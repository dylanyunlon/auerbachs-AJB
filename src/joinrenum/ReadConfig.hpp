using namespace std;
// [AJB] ReadConfig: db/{filenames,numlines,relations}.txt 解析器
// 输入格式:
//   filenames.txt: "R1 db/Ra.csv" per line
//   numlines.txt:  "R1 4" per line (0 = auto-count from file)
//   relations.txt: "R1(A,B)" per line

unordered_map<string, string> readFilenames(const string& filename){
    ifstream f;
    f.open(filename);
    assert(!f.fail()); //assert that opening file succeeded
    
    string tblName, fileName;
    
    unordered_map<string, string> mp;
    while(f >> tblName >> fileName) {
        mp[tblName] = fileName;
        fprintf(stderr, "[AJB_TRACE][ReadConfig] filename: %s -> %s\n", tblName.c_str(), fileName.c_str());
    }
    fprintf(stderr, "[AJB_STATE][ReadConfig] loaded %zu filename mappings from %s\n", mp.size(), filename.c_str());
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
    // [AJB_STATE] numlines=0 意味着运行时从文件计数, 会影响启动速度
    int zero_count = 0;
    for(auto& kv : mp) if(kv.second == 0) zero_count++;
    fprintf(stderr, "[AJB_STATE][ReadConfig] loaded %zu numlines from %s (%d need auto-count)\n",
            mp.size(), filename.c_str(), zero_count);
    return mp;
}

unordered_map<string, vector<string> > readRelations(const string& filename){
    ifstream f;
    f.open(filename);
    assert(!f.fail()); //assert that opening file succeeded
    
    unordered_map<string, vector<string> > mp;
    // --- parser: pointer-walk replacing repeated substr+find ---
    // upstream: line.find("(") → line.substr → while(line.find(",")) → line.substr
    //   creates many temporary std::string objects per line
    // changed: single scan with start/end pointers, extract directly
    //   into vector<string> — fewer allocations per relation
    string line;
    while(getline(f, line)){
        const char* p = line.c_str();
        const char* end = p + line.size();
        // find '(' — relation name is everything before it
        const char* paren = p;
        while(paren < end && *paren != '(') paren++;
        if(paren >= end) continue;  // malformed line
        string name(p, paren - p);
        // parse comma-separated attributes between '(' and ')'
        vector<string> columnName;
        const char* attr_start = paren + 1;
        for(const char* scan = attr_start; scan <= end; scan++) {
            if(scan == end || *scan == ',' || *scan == ')') {
                // trim leading/trailing spaces from attribute name
                const char* as = attr_start;
                const char* ae = scan;
                while(as < ae && *as == ' ') as++;
                while(ae > as && *(ae-1) == ' ') ae--;
                if(ae > as) columnName.emplace_back(as, ae - as);
                attr_start = scan + 1;
                if(*scan == ')') break;
            }
        }
        mp[name] = columnName;
        // [AJB_TRACE] 解析出的schema: 每个relation的属性列表
        fprintf(stderr, "[AJB_TRACE][ReadConfig] relation %s: arity=%zu attrs=[", name.c_str(), columnName.size());
        for(size_t i = 0; i < columnName.size(); i++){
            if(i) fprintf(stderr, ",");
            fprintf(stderr, "%s", columnName[i].c_str());
        }
        fprintf(stderr, "]\n");
    }
    fprintf(stderr, "[AJB_STATE][ReadConfig] loaded %zu relations from %s\n", mp.size(), filename.c_str());
    return mp;
}
