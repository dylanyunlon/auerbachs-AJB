#include<bits/stdc++.h>
using namespace std;
int main() {
    // read from db/Sampled.txt
    ifstream fin("db/Sampled.txt");
    ofstream fout("db/R1.txt");
    string line;
    while (getline(fin, line)) {
        // parse line "x y" and output to db/R1.txt as "x|y"
        stringstream ss(line);
        string x, y;
        ss >> x >> y;
        fout << x << "|" << y << endl;
    }
}