#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <tuple>
#include <chrono>
#include <vector>
#include <map>
#include <bitset>
#include <set>
#include <map>
#include <unordered_map>
#include <math.h>
#include <filesystem>
#include <algorithm>

#define TARGET_MEMORY_RATIO {1.25, 1.5, 2} // MUST be in increasing order
#define TARGET_DIM    {64}
#define INITIAL_MIN_RATIO 10000
#define DECREMENT_THRESHOLD 0.001

#ifndef EMBEDDING_DIM
#define EMBEDDING_DIM 64
#endif

#ifndef PARTITION_SIZE
#define PARTITION_SIZE 128
#endif

using namespace std;
using namespace std::chrono;
using namespace std::filesystem;

struct Args {
    string dataset_name;
    string option;
    double min_ratio;
    double memory_limit;
    int num_partition;
    int partition_size;
};

class MappingTable {
    map<int, vector<pair<int, set<int> > >, greater<int> > count_to_merged_sets;
    vector<vector<set<int> > > queries_per_cluster;
    vector<vector<int> > cluster_remap_table;
    vector<int> transaction_weight;
    vector<int> mapping_table;
    string filteredFilePrefix;
    string outputFilePrefix;
    string option;
    double min_ratio;
    int num_partition;
    int partition_size;
    int max_cluster_size;
    int final_table_size;

    void saveRemapTable() {
        cluster_remap_table.clear();
        vector<int> temp(partition_size, 0);
        for (int i=0; i<num_partition; i++) {
            cluster_remap_table.push_back(temp);
        }

        int remapped_cid = 0;
        for (auto ctm : count_to_merged_sets) {
            for (auto merged : ctm.second) {
                for (auto cid : merged.second) {    
                    cluster_remap_table[merged.first][cid] = remapped_cid;
                    remapped_cid++;
                }
            }
        }
        assert (remapped_cid == partition_size * num_partition);
    }

public:
    vector<int> EMBEDDING_DIM_LIST = TARGET_DIM;
    vector<float> MEMORY_RATIO_LIST = TARGET_MEMORY_RATIO;
    int num_total_feature;

    explicit MappingTable(Args &args) : num_partition(args.num_partition), partition_size(args.partition_size), min_ratio(args.min_ratio) {
        string homeDir = getenv("HOME");
        filteredFilePrefix = homeDir + "/MERCI/data/4_filtered/" + args.dataset_name + "/" + args.dataset_name;

        option = args.option;

        outputFilePrefix = homeDir + "/MERCI/data/6_evaluation_input/" + args.dataset_name + "/" + "partition_" + to_string(num_partition) + "/";

        create_directory(outputFilePrefix);
        max_cluster_size = 1;   // Initially, max cluster size is one
    }

    void ClusterIdentityRemap() {
        cluster_remap_table.clear();
        cluster_remap_table.resize(num_partition);

        int remapped_cid = 0;
        for (int i=0; i<num_partition; i++) {
            for (int j=0; j<partition_size; j++) {
                cluster_remap_table[i].push_back(remapped_cid);
                remapped_cid++;
            }
        }
        assert (remapped_cid == partition_size * num_partition);
    }

    void readTestMetaData() {
        string testFileName = filteredFilePrefix + "_test_filtered.txt";

        ifstream testFile;
        testFile.open(testFileName, ifstream::in);
        if (testFile.fail()) {
            cout << "Can not open: " << testFileName << endl;
            exit(EXIT_FAILURE);
        }

        char sharp;
        testFile >> sharp;
        assert(sharp == '#');

        testFile >> num_total_feature;
        cout << "Total # of features (train+test) : " << num_total_feature << endl << endl;
    }

    void readTrainQueries(vector<pair<int, int> > &feature_to_pos) {
        queries_per_cluster.resize(num_partition, vector<set<int> >(partition_size, set<int>()));

        string trainFileName = filteredFilePrefix + "_train_filtered.txt";

        ifstream trainFile;
        trainFile.open(trainFileName, ifstream::in);
        if (trainFile.fail()) {
            cout << "Can not open: " << trainFileName << endl;
            exit(EXIT_FAILURE);
        }

        cout << "Reading train file start." << endl;

        string line;
        getline(trainFile, line); // header
        int transaction_num = 0;
        while (getline(trainFile, line)) {
            if(line.empty())
                continue;
            int feature_id;
            int weight = 0;
            stringstream ss(line);
            ss >> weight;
            if (weight <= 0) {
                cout << "Convert ERROR: weight of train query <= 0." << endl;
                exit(EXIT_FAILURE);
            }

            while (ss >> feature_id) {
                auto [p_id, c_id] = feature_to_pos[feature_id];
                queries_per_cluster[p_id][c_id].insert(transaction_num);
            }

            transaction_weight.push_back(weight);
            transaction_num++;
        }

        cout << "Reading train file done." << endl << endl;
    }

    int accum_transaction_weight(vector<int>& transaction_weight, const set<int>& transaction){
        int weight_accum = 0;
        for (auto it:transaction)
            weight_accum += transaction_weight[it];
        return weight_accum;
    }

    void CorrelationAwareVariableSizedClustering(vector<pair<int, int> > &feature_to_pos, vector<vector<int> > &feature_id_per_partition, vector<vector<int> > &feature_cluster_per_partition, int num_train) {
        
        for (auto dim : EMBEDDING_DIM_LIST) {

            vector<set<set<int>>> empty_cluster_per_partition(num_partition, set<set<int>>());
            vector<set<pair<set<int>, set<int>>>> global_merge_table; // global_merge_table[pid] = set of pair<set of features in merged cluster, set of queries that belong to merged cluster>
            global_merge_table.resize(num_partition);
            vector<float> local_memories(num_partition);
            float global_memory = 0.;
            double min_ratio = INITIAL_MIN_RATIO;

            cout << "____________________________________________________________________" << endl;
            cout << "Constructing intial global merge table for dimension size " << dim << "B" << endl << endl;
            
            /* initial construction of merged_table and memory */
            #pragma omp parallel for
            for (auto pid=0; pid < num_partition; pid++) {

                const vector<set<int>> &queries = queries_per_cluster[pid]; // inverted index for each cluster: queries[c_id] = set of query ids

                for (int i=0;i<partition_size;i++) {
                    if (queries[i].size() != 0) {
                        set<int> cluster;
                        cluster.insert(i);
                        
                        pair<set<int>, set<int>> result = make_pair(cluster, queries[i]);
                        global_merge_table[pid].insert(result);

                        local_memories[pid] += double(4) * dim/1024/1024/1024; 
                    }
                    else {
                        set<int> dummy;
                        dummy.insert(i);
                        empty_cluster_per_partition[pid].insert(dummy);
                    }
                }
            }
            
            global_memory = accumulate(local_memories.begin(), local_memories.end(), 0.);
            cout << "Initial Memory for " << dim << "B dimension: " << global_memory << "GB" << endl << endl;

            for (auto target_memory_ratio : MEMORY_RATIO_LIST) {
                float target_memory = num_total_feature * dim * 4. * target_memory_ratio / 1024 / 1024 / 1024;
                cout << "____________________________________________________________________" << endl;
                cout << "Searching for the target memory ratio of " << target_memory_ratio;
                cout << "(" << target_memory <<  "GB, Dimension Size " << dim << "B)" << endl;

                double min_ratio_decrement = min_ratio / 5;
                vector<set<set<int>>> merged_info_per_partition(num_partition, set<set<int>>());
                count_to_merged_sets.clear();

                cout << endl << "Initiating parallel local merging" << endl;

                bool beginParallel = true;

                if (global_memory >= target_memory) {
                    if (abs(global_memory - target_memory) < 0.1 * target_memory) {
                        cout << "Target memory already achieved (" << global_memory << "GB); finishing parallel merging" << endl;
                        beginParallel = false;
                    } else {
                        cout << "Initial memory surpasses the target memory too much; change initial min_ratio and restart the program" << endl;
                        exit(1);
                    }
                }

                while (beginParallel) {
                    vector<set<pair<set<int>, set<int>>>> table_backup = global_merge_table;
                    vector<double> local_memory_increments(num_partition);

                    /* Locally find two clusters with greatest benefit/cost, and merge locally iteratively */
                    #pragma omp parallel for schedule(dynamic)
                    for (auto pid = 0; pid < num_partition; pid++) {

                        while (1) {
                            float local_max_ratio = -1.;
                            double candidate_cost = -1.;
                            pair<set<int>, set<int> > candidate_1;
                            pair<set<int>, set<int> > candidate_2;

                            /* Search for local maximum pair */
                            for (auto it = global_merge_table[pid].begin(); it != global_merge_table[pid].end(); it++) {
                                for (auto it2 = next(it, 1); it2 != global_merge_table[pid].end(); it2++) {
                                    // computing benenfit/cost for each possible pair
                                    set<int> intersect;
                                    std::set_intersection(it->second.begin(), it->second.end(), it2->second.begin(), it2->second.end(), std::inserter(intersect, intersect.begin()));    

                                    int weight_accum = accum_transaction_weight(transaction_weight, intersect); // benefit: weighted sum of number of queries that belong to both clusters
                                    float cost = pow(2, (it->first.size() + it2->first.size())) - pow(2, it->first.size()) - pow(2, it2->first.size()) + 1; // cost: 2^(a+b) - 2^a -2^b +1
                                    float ratio = weight_accum / cost;

                                    if (ratio > local_max_ratio && ratio > min_ratio) {
                                        local_max_ratio = ratio;
                                        candidate_1 = *it;
                                        candidate_2 = *it2;
                                        candidate_cost = cost * 4. * dim/1024/1024/1024;
                                    }
                                }
                            }

                            /* This partition is done if no pair found */
                            if (local_max_ratio == -1.)
                                break;

                            /* Create new merged set */
                            set <int> merged_set;
                            set <int> merged_transactions;
                            std::set_union(candidate_1.first.begin(), candidate_1.first.end(), candidate_2.first.begin(), candidate_2.first.end(), std::inserter(merged_set, merged_set.begin()));
                            std::set_union(candidate_1.second.begin(), candidate_1.second.end(), candidate_2.second.begin(), candidate_2.second.end(), std::inserter(merged_transactions, merged_transactions.begin()));
                            pair <set<int>, set<int>> result = make_pair(merged_set, merged_transactions);

                            // insert back merged cluster to merge table
                            global_merge_table[pid].erase(candidate_1);
                            global_merge_table[pid].erase(candidate_2);
                            global_merge_table[pid].insert(result);

                            // increment memory
                            local_memory_increments[pid] += candidate_cost;
                        }
                    }

                    /* terminate condition for parallel merging */
                    float global_memory_increment = accumulate(local_memory_increments.begin(), local_memory_increments.end(), 0.);
                    if (global_memory + global_memory_increment >= target_memory * 1.05) {
                        cout << "    Memory at min_ratio " << min_ratio << ": " << global_memory + global_memory_increment << "GB" << endl;
                        global_merge_table = table_backup;

                        min_ratio_decrement /= 2;
                        if (min_ratio_decrement < DECREMENT_THRESHOLD) {
                            cout << "    Memory Surpassed: Stopping parallel merge... rolling back merge table" << endl;
                            break; 
                        } else {
                            min_ratio += min_ratio_decrement;
                            cout << "    Memory Surpassed: Continuing parallel merge with new min_ratio decrement...";
                            cout << min_ratio_decrement << " rolling back merge table " << endl;
                            continue;
                        }
                    } 

                    global_memory += global_memory_increment;
                    for (auto pid = 0; pid < num_partition; pid++)
                        local_memories[pid] += local_memory_increments[pid];
                    cout << "    Memory at min_ratio " << min_ratio << ": " << global_memory << "GB" << endl;

                    if (global_memory >= target_memory * 0.95) {
                        cout << "    Memory in Bound: Stopping parallel merge" << endl;
                        break; 
                    }

                    min_ratio = min_ratio - min_ratio_decrement;
                    if (min_ratio == 0.0) {
                        min_ratio_decrement /= 2;
                        min_ratio += min_ratio_decrement;
                    }
                    cout << "    Continuing parallel merge with at the new min_ratio of " << min_ratio << endl;
                }

                cout << endl << "Initiating sequential global merging" << endl;

                /* Begin sequential merging */
                while (abs(global_memory - target_memory) >= 0.1 * target_memory) {
                    float global_max_ratio = -1.;
                    int candidate_pid = -1;
                    pair<set<int>, set<int> > candidate_1;
                    pair<set<int>, set<int> > candidate_2;
                    double candidate_cost = -1.;

                    /* Globally & Iteratively find two clusters with greatest benefit/cost to merge */
                    for (auto pid=0; pid < num_partition; pid++) {
                        for (auto it = global_merge_table[pid].begin(); it != global_merge_table[pid].end(); it++) {
                            for (auto it2 = next(it, 1); it2 != global_merge_table[pid].end(); it2++) {
                                // computing benenfit/cost for each possible pair
                                set<int> intersect;
                                std::set_intersection(it->second.begin(), it->second.end(), it2->second.begin(), it2->second.end(), std::inserter(intersect, intersect.begin()));    

                                int weight_accum = accum_transaction_weight(transaction_weight, intersect); // benefit: weighted sum of number of transactions that belong to both clusters
                                float cost = pow(2, (it->first.size() + it2->first.size())) - pow(2, it->first.size()) - pow(2, it2->first.size()) + 1; // cost: 2^(a+b) - 2^a -2^b +1
                                float ratio = weight_accum / cost;

                                if (ratio > global_max_ratio) {
                                    global_max_ratio = ratio;
                                    candidate_pid = pid;
                                    candidate_1 = *it;
                                    candidate_2 = *it2;
                                    candidate_cost = cost * 4. * dim/1024/1024/1024;
                                }
                            }
                        }
                    }

                    /* terminate condition; do not merge selected candidates */
                    if (global_memory + candidate_cost > target_memory)
                        break;
                        
                    /* Update merge table of selected pid */
                    else {
                        // remove selected candidates from merge table
                        global_merge_table[candidate_pid].erase(candidate_1);
                        global_merge_table[candidate_pid].erase(candidate_2);

                        // create new merged set
                        set <int> merged_set;
                        set <int> merged_transactions;
                        std::set_union(candidate_1.first.begin(), candidate_1.first.end(), candidate_2.first.begin(), candidate_2.first.end(), std::inserter(merged_set, merged_set.begin()));
                        std::set_union(candidate_1.second.begin(), candidate_1.second.end(), candidate_2.second.begin(), candidate_2.second.end(), std::inserter(merged_transactions, merged_transactions.begin()));


                        pair <set<int>, set<int>> result = make_pair(merged_set, merged_transactions);

                        // insert back merged cluster to merge table
                        global_merge_table[candidate_pid].insert(result);

                        // increment memory
                        global_memory += candidate_cost;
                    }

                    cout << "    Current memory : " << global_memory << "GB" << endl;;
                }

                /* store merge results to merged_info_per_partition */
                for (auto pid=0; pid < num_partition; pid++) {
                    for (auto cluster : global_merge_table[pid]) {
                        merged_info_per_partition[pid].insert(cluster.first);
                    }
                }

                double total_memory = 0;
                for(auto pid=0; pid<num_partition; pid++) {
                    int size_accum = 0;
                    set<set<int> > &merged_info = merged_info_per_partition[pid];

                    for (auto it: merged_info) {
                        total_memory += pow(2, it.size())*4*dim/1024/1024/1024;
                        size_accum += it.size();
                        count_to_merged_sets[it.size()].push_back(make_pair(pid, it));
                    }
                    for (auto it: empty_cluster_per_partition[pid]) {
                        size_accum += it.size();
                        count_to_merged_sets[it.size()].push_back(make_pair(pid, it));
                    }
                    assert (size_accum == partition_size);
                }

                cout << endl << "Total memory : " << total_memory << "GB" << endl << endl;
                cout << "Saving result..." << endl;

                saveRemapTable();

                remapfeatureID(feature_id_per_partition, feature_cluster_per_partition, num_train);
                writeRemappedTestFile(dim, target_memory_ratio);

                cout << "Saved!" << endl << endl;
            }
        }
    }

    vector<vector<int> > saveRemappedQuery() {
        string testFileName = filteredFilePrefix + "_test_filtered.txt";
        
        ifstream testFile; 
        testFile.open(testFileName, ifstream::in);
        if (testFile.fail()) {
            cout << "Can not open: " << testFileName << endl;
            exit(EXIT_FAILURE);
        }

        vector<vector<int> > query;
        int num_query_weighted = 0;

        string line;
        getline(testFile, line); // header
        while (getline(testFile, line)) {
            if (line.empty()) continue;

            vector<int> single_query;
            int feature_id;
            int weight = 0;
            stringstream ss(line);

            ss >> weight;
            if (weight <= 0) {
                cout << "Weight of test query <= 0." << endl;
                exit(EXIT_FAILURE);
            }
            num_query_weighted += weight;

            while (ss >> feature_id) {
                single_query.push_back(mapping_table[feature_id]);
            }
            sort(single_query.begin(), single_query.end());

            for (int i=0; i<weight; i++) {
                query.push_back(single_query);
            }
        }

        random_shuffle(query.begin(), query.end());
        testFile.close();
        cout << "Processed " << num_query_weighted << " transactions." << endl << endl;

        return query;
    }

    void writeRemappedTestFile(int dim = EMBEDDING_DIM, float mem = 0.) {
        vector<vector<int> > query = saveRemappedQuery();
        string outputFileName;

        if (option=="clustering") outputFileName = outputFilePrefix + "test_" + to_string(mem) + "X_" + to_string(dim) + "dim_" + to_string(partition_size) + ".dat";
        else if (option == "remap-only") outputFileName = outputFilePrefix + "test_remap_only.dat";

        ofstream outFile; 
        outFile.open(outputFileName, ofstream::out);
        if (outFile.fail()) {
            cout << "Can not open: " << outputFileName << endl;
            exit(EXIT_FAILURE);
        }
        outFile << "# " << num_total_feature << " " << final_table_size << " ";

        int offset;
        if (option == "remap-only") {
            offset = partition_size * num_partition;
            outFile << "# 2 0 1 " << offset << " ";
        }
        else {
            offset = 0;
            int accum_count = 0;

            outFile << "# " << count_to_merged_sets.size()+1 << " ";
            for (auto c : count_to_merged_sets) {
                accum_count += c.second.size();
                outFile << offset << " " << max_cluster_size * c.first << " " << accum_count << " ";
                offset += (max_cluster_size * c.first * c.second.size());
            } 
        }

        assert (offset == partition_size * num_partition);

        outFile << offset << " -1 -1" << endl;

        for (int q_idx=0; q_idx<int(query.size()); q_idx++) {
            int feature_idx;
            for (feature_idx=0; feature_idx<int(query[q_idx].size())-1; feature_idx++) {
                outFile << query[q_idx][feature_idx] << " ";
            }
            outFile << query[q_idx][feature_idx] << endl;
        }

        outFile.close();
        cout << "Remapped test file done." << endl << endl;
    }

    void remapfeatureID(vector<vector<int> > &feature_id_per_partition, vector<vector<int> > &feature_cluster_per_partition, int num_train) {
        mapping_table.clear();
        mapping_table.resize(num_total_feature+1, -1);

        vector<int> start_index(num_partition*partition_size, 0);
        for (int p_id=0; p_id<num_partition; p_id++) {
            for (int feature_idx=0; feature_idx<int(feature_id_per_partition[p_id].size()); feature_idx++) {
                int remapped_cluster = cluster_remap_table[p_id][feature_cluster_per_partition[p_id][feature_idx]];
                int remapped_id = remapped_cluster * max_cluster_size + start_index[remapped_cluster];
                mapping_table[feature_id_per_partition[p_id][feature_idx]] = remapped_id;
                start_index[remapped_cluster]++;
            }
        }

        //left ids
        int mapping_table_size = max_cluster_size * partition_size * num_partition;
        final_table_size = mapping_table_size;
        for (int feature_id=1; feature_id<num_total_feature+1; feature_id++) {
            if (mapping_table[feature_id] == -1) {
                assert (feature_id > num_train);
                mapping_table[feature_id] = final_table_size;
                final_table_size++;
            }
        }
        cout << "Final mapping table size: " << final_table_size << endl << endl;;
    }
};

class ClusterInPartition {
    vector<vector<int> > feature_count_per_cluster;
    string dataset_name;
    int max_feature_per_partition;
    int num_train_feature_in_remap;
    int num_partition;
    int partition_size;

    void printSummary() {
        cout << "\n==== CLUSTER IN PARTITION META INFO ===" << endl;
        cout << "# of features (train)            : " << num_train_feature << endl;
        cout << "# of partitions                  : " << num_partition << endl;
        cout << "Size of a partition              : " << partition_size << endl;
        cout << "Max # of features in a partition : " << max_feature_per_partition << endl;
        cout << "=========================================" << endl << endl;
    }

    void readPaToH() {
        string homeDir = getenv("HOME");
        string patohFileName = homeDir + "/MERCI/data/5_patoh/" + dataset_name + "/partition_" + to_string(num_partition) + "/" + dataset_name + "_train_filtered.txt.part." + to_string(num_partition);
        ifstream patohFile;
        patohFile.open(patohFileName, ifstream::in);
        if (patohFile.fail()) {
            cout << "Can not open: " << patohFileName << endl;
            exit(EXIT_FAILURE);
        }
        int pid;
        int feature_id = 1;
        while (patohFile >> pid) {
            feature_count_per_partition[pid]++;
            feature_id_per_partition[pid].push_back(feature_id);
            feature_id++;
        }
        patohFile.close();

        num_train_feature = feature_id-1;

        // Note that initially all features belong to individual clusters (one feature in one cluster)
        for (int i=0; i<num_partition; i++) {
            for (int j=0; j<(int)feature_id_per_partition[i].size(); j++) {
                feature_cluster_per_partition[i].push_back(j);
            }
        }

        max_feature_per_partition = *max_element(feature_count_per_partition.begin(), feature_count_per_partition.end());

        assert (max_feature_per_partition <= PARTITION_SIZE);
    }

    void saveRemapFile(ifstream &remapFile, int partition_id) {
        int feature_id;
        while (remapFile >> feature_id) {
            feature_id_per_partition[partition_id].push_back(feature_id);
            num_train_feature_in_remap++;
        }
    }

public:
    vector<vector<int> > feature_cluster_per_partition; // Tracks which cluster each belons to [Partition x Features in Partition]
    vector<vector<int> > feature_id_per_partition;
    vector<int> feature_count_per_partition;
    int num_train_feature;

    explicit ClusterInPartition(Args &args) : num_partition(args.num_partition), dataset_name(args.dataset_name), partition_size(args.partition_size) {
        feature_count_per_cluster.resize(num_partition);
        for (int p_id=0; p_id<num_partition; p_id++) {
            feature_count_per_cluster[p_id].resize(partition_size);
        }
        feature_cluster_per_partition.resize(num_partition);
        feature_id_per_partition.resize(num_partition);
        feature_count_per_partition.resize(num_partition);

        num_train_feature = 0;
        num_train_feature_in_remap = 0;
    }

    void setup_for_clustering() {
        readPaToH();
        printSummary();
        // Release memory.
        vector<vector<int> >().swap(feature_count_per_cluster);
    }

    vector<pair<int, int> > getfeatureToPosInt() {
        vector<pair<int, int> > feature_to_pos(num_train_feature+1);
        for (int p_id=0; p_id<num_partition; p_id++) {
            for (int feature_idx=0; feature_idx<int(feature_id_per_partition[p_id].size()); feature_idx++) {
                int feature_id = feature_id_per_partition[p_id][feature_idx];
                int c_id = feature_cluster_per_partition[p_id][feature_idx];
                feature_to_pos[feature_id] = make_pair(p_id, c_id);
            }
        }

        return feature_to_pos;
    }
};

Args readArguments(int argc, const char *argv[]) {
    Args args;
    if((argc != 5) && (argc != 6)) {
        cout << "Usage 1 : ./bin/clustering -d <dataset name> -p <# of partitions>" << endl;
        cout << "Usage 2 : ./bin/clustering -d <dataset name> -p <# of partitions> --remap-only" << endl;
        exit(EXIT_FAILURE);
    }

    args.option = "clustering";
    args.partition_size = PARTITION_SIZE;

    for (int i=1; i < argc; i++) {
        string arg(argv[i]);

        if (arg.compare("-d") == 0 || arg.compare("--dataset") == 0) {
            args.dataset_name = string(argv[i+1]);
            ++i;
        }
        else if (arg.compare("-p") == 0 || arg.compare("--num_partition") == 0) {
            args.num_partition = stoi(argv[i+1]);
            ++i;
        }
        else if (arg.compare("--remap-only") == 0) {
            args.option = "remap-only";
        }
        else {
            cout << "Wrong argument option" << endl;
            exit(EXIT_FAILURE);
        }
    }
    cout << "Converting " << args.dataset_name << " / Partition " << args.num_partition << " / Size of Partition " << args.partition_size << ".\n\n";  
    return args;
}

int main(int argc, const char *argv[]) {

    // Parse Arguments.
    Args args = readArguments(argc, argv);
    // Read feature id & cluster id.
    ClusterInPartition CIP(args);
    CIP.setup_for_clustering();

    // Get feature to position <partition id, cluster id> mapping (1-index).
    vector<pair<int, int> > feature_to_pos = CIP.getfeatureToPosInt();

    // Read train file to decide which clusters to merge.
    MappingTable table(args);
    table.readTestMetaData();
    table.readTrainQueries(feature_to_pos);

    if (args.option == "clustering") {
        table.CorrelationAwareVariableSizedClustering(feature_to_pos, CIP.feature_id_per_partition, CIP.feature_cluster_per_partition, CIP.num_train_feature);
    }
    else if (args.option == "remap-only") {
        table.ClusterIdentityRemap();
        table.remapfeatureID(CIP.feature_id_per_partition, CIP.feature_cluster_per_partition, CIP.num_train_feature);
        table.writeRemappedTestFile();
    }
    else {
        assert (false);
    }

    return 0;
}

