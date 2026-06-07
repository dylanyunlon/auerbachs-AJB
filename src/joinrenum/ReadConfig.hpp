using namespace std;
// [AJB] ReadConfig: db/{filenames,numlines,relations}.txt 解析器
// 输入格式:
//   filenames.txt: "R1 db/Ra.csv" per line
//   numlines.txt:  "R1 4" per line (0 = auto-count from file)
//   relations.txt: "R1(A,B)" per line

// [AJB] ReadConfig diagnostics
#include <cstdio>
#include <numeric>
#include <set>
static thread_local struct {
    long long files_parsed = 0;
    long long numlines_parsed = 0;
    long long relations_parsed = 0;
    long long total_attributes = 0;    // across all relations
    int       max_arity = 0;           // max columns in a single relation
    int       min_arity = 9999;        // min columns in a single relation
    long long total_rows = 0;          // sum of all numlines
    long long auto_count_tables = 0;   // tables with numlines=0
    long long malformed_lines = 0;     // lines that couldn't be parsed
    // [AJB_BP] M936: attribute frequency — how often each attr appears across relations
    // (tracked via dump, not stored here to keep struct simple)
    void dump(const char* tag = "ReadConfig") {
        fprintf(stderr, "[AJB_STATE][%s] files=%lld numlines=%lld relations=%lld\n",
                tag, files_parsed, numlines_parsed, relations_parsed);
        fprintf(stderr, "[AJB_STATE][%s] total_attrs=%lld arity_range=[%d,%d] avg_arity=%.1f\n",
                tag, total_attributes, min_arity == 9999 ? 0 : min_arity, max_arity,
                relations_parsed > 0 ? (double)total_attributes / relations_parsed : 0.0);
        fprintf(stderr, "[AJB_STATE][%s] total_rows=%lld auto_count=%lld malformed=%lld\n",
                tag, total_rows, auto_count_tables, malformed_lines);
    }
    void reset() {
        files_parsed = numlines_parsed = relations_parsed = total_attributes = 0;
        max_arity = 0; min_arity = 9999;
        total_rows = auto_count_tables = malformed_lines = 0;
    }
} ajb_rc_stats;

unordered_map<string, string> readFilenames(const string& filename){
    ifstream f;
    f.open(filename);
    assert(!f.fail()); //assert that opening file succeeded
    
    string tblName, fileName;
    
    unordered_map<string, string> mp;
    // [AJB_BP] M936: track file extensions to detect data format
    std::set<string> extensions;
    while(f >> tblName >> fileName) {
        mp[tblName] = fileName;
        ajb_rc_stats.files_parsed++;
        // [AJB_BP] M936: extract extension
        size_t dot = fileName.rfind('.');
        if (dot != string::npos) extensions.insert(fileName.substr(dot));
        fprintf(stderr, "[AJB_TRACE][ReadConfig] filename: %s -> %s\n", tblName.c_str(), fileName.c_str());
    }
    // [AJB_STATE] M936: summary with extension distribution
    fprintf(stderr, "[AJB_STATE][ReadConfig] loaded %zu filename mappings from %s\n", mp.size(), filename.c_str());
    fprintf(stderr, "[AJB_STATE][ReadConfig] file extensions: ");
    for (auto& ext : extensions) fprintf(stderr, "%s ", ext.c_str());
    fprintf(stderr, "\n");
    return mp;
}

unordered_map<string, int> readNumLines(const string& filename){
    ifstream f;
    f.open(filename);
    assert(!f.fail()); //assert that opening file succeeded
    
    string tblName;
    int numlines;
    
    unordered_map<string, int> mp;
    int max_rows = 0, min_rows = INT_MAX;
    while(f >> tblName >> numlines) {
        mp[tblName] = numlines;
        ajb_rc_stats.numlines_parsed++;
        ajb_rc_stats.total_rows += numlines;
        if (numlines == 0) ajb_rc_stats.auto_count_tables++;
        if (numlines > max_rows) max_rows = numlines;
        if (numlines < min_rows) min_rows = numlines;
        // [AJB_TRACE] M937: print each table's row count
        fprintf(stderr, "[AJB_TRACE][ReadConfig] numlines: %s = %d%s\n",
                tblName.c_str(), numlines, numlines == 0 ? " (AUTO)" : "");
    }
    // [AJB_STATE] numlines=0 意味着运行时从文件计数, 会影响启动速度
    int zero_count = 0;
    for(auto& kv : mp) if(kv.second == 0) zero_count++;
    fprintf(stderr, "[AJB_STATE][ReadConfig] loaded %zu numlines from %s (%d need auto-count)\n",
            mp.size(), filename.c_str(), zero_count);
    // [AJB_STATE] M937: row distribution summary
    fprintf(stderr, "[AJB_STATE][ReadConfig] rows: total=%lld min=%d max=%d avg=%.0f\n",
            ajb_rc_stats.total_rows,
            mp.empty() ? 0 : min_rows,
            mp.empty() ? 0 : max_rows,
            mp.empty() ? 0.0 : (double)ajb_rc_stats.total_rows / mp.size());
    // [AJB_BP] M937: warn about skew (max/min ratio > 100)
    if (!mp.empty() && min_rows > 0 && max_rows / min_rows > 100)
        fprintf(stderr, "[AJB_BP][ReadConfig] WARNING: high row skew ratio=%d (max=%d min=%d)\n",
                max_rows / min_rows, max_rows, min_rows);
    return mp;
}

unordered_map<string, vector<string> > readRelations(const string& filename){
    ifstream f;
    f.open(filename);
    assert(!f.fail()); //assert that opening file succeeded
    
    unordered_map<string, vector<string> > mp;
    // [AJB_BP] M938: attribute frequency map for join-graph connectivity
    unordered_map<string, int> attr_freq;
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
        if(paren >= end) {
            ajb_rc_stats.malformed_lines++;
            fprintf(stderr, "[AJB_BP][ReadConfig] malformed line (no paren): %s\n", line.c_str());
            continue;  // malformed line
        }
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
        ajb_rc_stats.relations_parsed++;
        ajb_rc_stats.total_attributes += columnName.size();
        int arity = (int)columnName.size();
        if (arity > ajb_rc_stats.max_arity) ajb_rc_stats.max_arity = arity;
        if (arity < ajb_rc_stats.min_arity) ajb_rc_stats.min_arity = arity;
        // [AJB_BP] M938: accumulate attribute frequency
        for (auto& a : columnName) attr_freq[a]++;
        // [AJB_TRACE] 解析出的schema: 每个relation的属性列表
        fprintf(stderr, "[AJB_TRACE][ReadConfig] relation %s: arity=%zu attrs=[", name.c_str(), columnName.size());
        for(size_t i = 0; i < columnName.size(); i++){
            if(i) fprintf(stderr, ",");
            fprintf(stderr, "%s", columnName[i].c_str());
        }
        fprintf(stderr, "]\n");
    }
    fprintf(stderr, "[AJB_STATE][ReadConfig] loaded %zu relations from %s\n", mp.size(), filename.c_str());
    // [AJB_STATE] M938: attribute frequency analysis for join connectivity
    int shared_attrs = 0, unique_attrs = 0;
    fprintf(stderr, "[AJB_STATE][ReadConfig] attribute frequency map (%zu distinct attrs):\n", attr_freq.size());
    for (auto& kv : attr_freq) {
        fprintf(stderr, "[AJB_TRACE][ReadConfig]   attr '%s' appears in %d relations%s\n",
                kv.first.c_str(), kv.second, kv.second > 1 ? " (JOIN ATTR)" : "");
        if (kv.second > 1) shared_attrs++;
        else unique_attrs++;
    }
    fprintf(stderr, "[AJB_STATE][ReadConfig] join graph: %d shared (join) attrs, %d unique attrs\n",
            shared_attrs, unique_attrs);
    // [AJB_STATE] M938: dump overall config stats
    ajb_rc_stats.dump();
    return mp;
}
