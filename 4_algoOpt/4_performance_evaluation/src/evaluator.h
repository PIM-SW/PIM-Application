
class Evaluator {
public:
    /*class members */
    vector<array<float, EMBEDDING_DIM>> embedding_table;    //embedding table
    vector<array<float, EMBEDDING_DIM>> qres;
    vector<TInfo> thrinfo;
    CustomBarrier cstart;
    CustomBarrier cend;
    
    size_t core_count;
    int num_features;
    int num_device;

    /* constructors */
    Evaluator(int num_features_, int core_count_) : num_features(num_features_), core_count(core_count_) {
        //allocate space for thrinfo
        thrinfo.resize(core_count);
        num_device = num_features/DEVICE_SIZE +1;
        for (size_t i=0; i<core_count; i++) {
            thrinfo[i].num_load_host = 0;
            thrinfo[i].num_load_pim = 0;
        }
    }

    void build_embedding_table() {
        //random initialization of embedding_table
        embedding_table.resize(num_features);
        for (int i=0; i < num_features; i++) {
            for (int j=0; j < EMBEDDING_DIM; j++)
                embedding_table[i][j] = 0.000001 * j;
        }
    }

    /* zero initialization of res vectors */
    void init(int qcount) {
        if(qres.size() == 0)
            qres.resize(qcount);
        for(int i=0; i<qcount; i++) {
            for(int j=0; j< EMBEDDING_DIM; j++){
                qres[i][j] = 0.0f;
            }
        }
    }
    void setqbase(vector<vector<vector<int>>> &partitioned_query) {
        int accum = 0;
        for(size_t i=0; i<partitioned_query.size(); i++) {
            thrinfo[i].qbase = accum;
            accum += partitioned_query[i].size();
        }
    }
};

class Baseline: public Evaluator {
public:
    /* constructors */
    Baseline(int num_features_, int core_count_) : Evaluator(num_features_, core_count_) {}

    /* Baseline process */
    void inline reduce(const vector< vector<int> > &query, const int core_id) {
        size_t qlen = query.size();
        int myqbase = thrinfo[core_id].qbase;
        for(size_t i=0; i < qlen; i++) {
            size_t curqlen = query[i].size();
            vector<int> pim_cnt(num_device, 0);
            int max_load_pim = 0;

            for(size_t j=0; j<curqlen; j++) {
                pim_cnt[query[i][j] / DEVICE_SIZE] += 1;
                for(int l=0; l < EMBEDDING_DIM; l++) {
                    qres[i + myqbase][l] += embedding_table[query[i][j]][l];
                }
            }

            for (int j=0; j<num_device; j++) {
                if (pim_cnt[j] > 0) {
                    thrinfo[core_id].num_load_host += 1;
                    if (pim_cnt[j] > max_load_pim) {
                        max_load_pim = pim_cnt[j];
                    }
                }
            }
            thrinfo[core_id].num_load_pim += max_load_pim;
        }
    }
    void process(const vector< vector<int> > &query, const int core_id) { 
        // Prologue
        cstart.barrier_wait(core_count);
        // Core
        reduce(query, core_id);
        // Epilogue
        cend.barrier_wait(core_count);
    }
};
