#include "utils.h"
#include "evaluator.h"

int main(int argc, const char *argv[]) {
    if (argc != 3) {
        cout << "Usage 1: ./bin/eval_baseline -d <dataset name>" << endl;
        return -1;
    }

    /* static file I/O variables */
    string homeDir = getenv("HOME");                        // home directory
    string merciDir = homeDir + "/MERCI/data/";                  // MERCI directory (located in $HOME/MERCI)
    string datasetDir = merciDir + "4_filtered";            // input dataset directory

    /*file I/O variables */
    string datasetName;
    string testFileName;
    ifstream testFile;

    /* counter variables */
    int num_features = 0;                   // total number of features (train + test)
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
    testFileName = datasetDir + "/"+ datasetName + "/" + datasetName + "_test_filtered.txt";
    cout << "testFilename : " << testFileName << endl;
    testFile.open(testFileName, ifstream::in);
    if (testFile.fail()) {
        cout << "FILE ERROR: Could not open test.dat file" << endl;
        return -1;
    }

    //Step 2: Read meta data
    testFile >> read_sharp;
    testFile >> num_features;

    cout << "============= META INFO =============" << endl;
    cout << "# of Features (train + test)    : " << num_features << endl;
    cout << "=====================================" << endl << endl;    

    //Step 3: Read and store query(test) transactions
    QueryData qd(testFile, true);
    testFile.close(); //close file

    vector<thread> t;
    t.resize(core_count);
    qd.partition(core_count);
    cout << endl;

    ///////////////////////////////////////////////////////////////////////////// 
    cout << "Eval Phase 2: Building Embedding Table..." << endl << endl;
    ///////////////////////////////////////////////////////////////////////////// 
    Baseline baseline(num_features, core_count);
    baseline.build_embedding_table();

    ///////////////////////////////////////////////////////////////////////////// 
    cout << "Eval Phase 3: Running Baseline..." << endl << endl;
    /////////////////////////////////////////////////////////////////////////////     
    eval<Baseline>(baseline, qd, core_count, t, repeat, "Baseline");

    return 0;
}
