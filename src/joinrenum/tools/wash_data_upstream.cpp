#include<bits/stdc++.h>
using namespace std;
int main() {
    ifstream fin("db/Sampled.txt");
    ofstream fout("db/R1.txt");
    string line;
    int converted = 0;
    while (getline(fin, line)) {
        // --- parser: find-based split replacing stringstream ---
        // upstream: stringstream ss(line); ss >> x >> y;
        // changed: find first whitespace, split into two substrings
        //   handles both space and tab delimiters, skips empty lines
        if(line.empty()) continue;
        size_t sep = line.find_first_of(" \t");
        if(sep == string::npos) continue;  // no delimiter found, skip
        string x = line.substr(0, sep);
        // skip past all whitespace between fields
        size_t next = line.find_first_not_of(" \t", sep);
        if(next == string::npos) continue;  // no second field
        // second field runs to end of line (trim trailing whitespace)
        size_t end = line.find_last_not_of(" \t\r\n");
        string y = line.substr(next, end - next + 1);
        fout << x << "|" << y << "\n";
        converted++;
    }
    if(converted == 0)
        cout << "Warning: no lines converted" << endl;
}
