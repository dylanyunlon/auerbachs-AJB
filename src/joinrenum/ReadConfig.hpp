using namespace std;
unordered_map<string, string> readFilenames(const string& filename){
    ifstream f;
    f.open(filename);
    assert(!f.fail()); //assert that opening file succeeded
    
    string tblName, fileName;
    
    unordered_map<string, string> mp;
    while(f >> tblName >> fileName) {
        mp[tblName] = fileName;
    }
    return mp;
}

unordered_map<string, int> readNumLines(const string& filename){
    ifstream f;
    f.open(filename);
    assert(!f.fail()); //assert that opening file succeeded
    
    string tblName;
    int numlines;
    
    unordered_map<string, int> mp;
    while(f >> tblName >> numlines) {
        mp[tblName] = numlines;
    }
    return mp;
}

unordered_map<string, vector<string> > readRelations(const string& filename){
    ifstream f;
    f.open(filename);
    assert(!f.fail()); //assert that opening file succeeded
    
    unordered_map<string, vector<string> > mp;
    // get each line of f and parse each line like "R(A, B, C)"
    string line;
    while(getline(f, line)){
        size_t pos = line.find("(");
        string name = line.substr(0, pos);
        vector<string> columnName = {};
        line = line.substr(pos + 1);
        while ((pos = line.find(",")) != string::npos) {
            columnName.push_back(line.substr(0, pos));
            line = line.substr(pos + 1);
        }
        columnName.push_back(line.substr(0, line.size() - 1));
        mp[name] = columnName;
    }
    return mp;
}
