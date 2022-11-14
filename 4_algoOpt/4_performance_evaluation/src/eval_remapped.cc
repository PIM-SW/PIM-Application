#include "utils.h"
#include "evaluator.h"

int main(int argc, const char *argv[]) {

    if (argc != 5) {
        cout << "Usage: ./bin/eval_remapped -d <dataset name> -p <#of partitions>" << endl;
        return -1;
    }

    /* static file I/O variables */
    string homeDir = getenv("HOME");                        // home directory
    string merciDir = homeDir + "/MERCI/data/";                  // MERCI directory (located in $HOME/MERCI)
    string datasetDir = merciDir + "6_evaluation_input/";   // input dataset directory

    /*file I/O variables */
    string datasetName;
    string testFileName;
    ifstream testFile;

    /* counter variables */
    int num_partition = 0;
    int num_features = 0;                   //total number of items (train + test)
    size_t core_count = thread::hardware_concurrency();
    int repeat = 1;

    /* helper variables */
    char read_sharp;                    //to read #

    /////////////////////////////////////////////////////////////////////////////  
    cout << "Eval Phase 0: Reading Command Line Arguments..." << endl << endl;
    /////////////////////////////////////////////////////////////////////////////

    /* parsing command line arguments */
    for (int i = 1; i < argc; i++) {
        string arg(argv[i]);

        if (arg.compare("-d") == 0 || arg.compare("--dataset") == 0) {
            datasetName = string(argv[i+1]);
        }
        else if (arg.compare("-p") == 0 || arg.compare("--num_partition") == 0) {
            num_partition = stoi(argv[i+1]);
        }
        else {
            cout << "ARG ERROR: Invalid Flag" << endl;
        }
        ++i; //skip next iteration
    }
    
    cout << "=================== EVAL INFO ==================" << endl;
    cout << "Embedding Dimension                   : " << EMBEDDING_DIM << endl;   
    cout << "Number of Threads                     : " << core_count << endl;
    cout << "===============================================" << endl << endl;    

    ///////////////////////////////////////////////////////////////////////////// 
    cout << "Eval Phase 1: Retrieving Test Queries..." << endl << endl;
    ///////////////////////////////////////////////////////////////////////////// 

    //Step 1: Open test.dat file
    testFileName = datasetDir + datasetName + "/partition_" + to_string(num_partition) + "/test_remap_only.dat";
    cout << "testFilename : " << testFileName << endl;
    testFile.open(testFileName, ifstream::in);
    if (testFile.fail()) {
        cout << "FILE ERROR: Could not open test.dat file" << endl;
        return -1;
    }

    //Step 2: Read meta data
    string line;
    getline(testFile, line);
    stringstream ss(line);
    ss >> read_sharp;
    ss >> num_features;

    cout << "============= META INFO =============" << endl;
    cout << "# of Items (train + test)    : " << num_features << endl;
    cout << "=====================================" << endl << endl;    

    //Step 3: Read and store query(test) transactions
    QueryData qd(testFile);
    testFile.close(); //close file

    vector<thread> t;
    t.resize(core_count);
    qd.partition(core_count);
    cout << endl;

    ///////////////////////////////////////////////////////////////////////////// 
    cout << "Eval Phase 2: Building Embedding Table..." << endl << endl;
    ///////////////////////////////////////////////////////////////////////////// 
    Baseline remapped(num_features+20000, core_count); // compensate for offsets in ID remapping
    remapped.build_embedding_table();

    ///////////////////////////////////////////////////////////////////////////// 
    cout << "Eval Phase 3: Running Remapped..." << endl << endl;
    /////////////////////////////////////////////////////////////////////////////     
    eval<Baseline>(remapped, qd, core_count, t, repeat, "Remapped");

    return 0;
}
