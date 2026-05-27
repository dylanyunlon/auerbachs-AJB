#include <boost/unordered_map.hpp>
using namespace std;

class JoinTree {
private:
    vector<CountOracle<int>*> CO;
    vector<vector<int> > relation; // relations[i][j] = the j-th attribute of the i-th relation
    vector<vector<int> > children; // children[i] = list of children of node i
    vector<int> parent; // parent[i] = parent of node i, -1 if root
    vector<vector<vector<int> > > joinPos; // joinPos[i][j] = the join positions of the edge between i and j
    vector<bool> visVar;
    vector<boost::unordered_map<vector<int>, int> > cache;
    vector<vector<int> > treeBound;

    void buildLeaves(int node, int fa = -1, int k = -1) {
        if(node < 0 || node >= (int)children.size()) return;
        bool flag = true;
        if(children[node].size() == 0) {
            if(fa == -1) return;
            vector<int> joinVals(joinPos[fa][k].size());
            for(int j = 0; j < joinPos[fa][k].size(); j++) {
                joinVals[j] = CO[node]->points[0][j];
            }
            cache[node][joinVals] = 1;
            for(int i = 1; i < CO[node]->points.size(); i++) {
                flag = true;
                for(int j = 0; j < joinPos[fa][k].size(); j++) {
                    joinVals[j] = CO[node]->points[i][j];
                    if(joinVals[j] != CO[node]->points[i - 1][j]) flag = false;
                }
                if(flag) {
                    cache[node][joinVals]++;
                }
                else cache[node][joinVals] = 1;
            }
            
            for(int i = 0; i < CO[node]->points.size(); i++){
                if(i > 0) CO[node]->points[i].cnt += CO[node]->points[i - 1].cnt;
            }
            return;
        }
        for(int i = 0; i < children[node].size(); i++) buildLeaves(children[node][i], node, i);
    }

    void preProcessing(int node, int fa = -1, int k = -1) {
        if(node < 0 || node >= (int)children.size() || children[node].size() == 0) return;
        for(int i = 0; i < children[node].size(); i++) preProcessing(children[node][i], node, i);
        vector<int> joinVals;
        for(int j = 0; j < children[node].size(); j++) {
            joinVals.resize(joinPos[node][j].size());
            for(int i = 0; i < CO[node]->points.size(); i++) {
                for(int k = 0; k < joinPos[node][j].size(); k++) joinVals[k] = CO[node]->points[i][joinPos[node][j][k]];
                CO[node]->points[i].cnt *= cache[children[node][j]][joinVals];
                // cout << "CO[node]->points[i].cnt: " << CO[node]->points[i].cnt << endl;
                // CO[node]->points[i].cnt *= CO[children[node][j]]->sumCnt(Point<int>(joinVals), Point<int>(joinVals));
            }
        }
        for(int i = 0; i < CO[node]->points.size(); i++){
            if(i > 0) CO[node]->points[i].cnt += CO[node]->points[i - 1].cnt;
        }
        if(fa == -1) return;
        bool flag = true;
        joinVals.resize(joinPos[fa][k].size());
        for(int i = 1; i < CO[node]->points.size(); i++) {
            flag = true;
            for(int j = 0; j < joinPos[fa][k].size(); j++) {
                joinVals[j] = CO[node]->points[i][j];
                if(joinVals[j] != CO[node]->points[i - 1][j]) flag = false;
            }
            if(flag) {
                cache[node][joinVals]++;
            }
            else cache[node][joinVals] = 1;
        }
        
        return;
    }

    void initCountRels(int node) {
        if(node < 0 || node >= (int)children.size()) return;
        vector<bool> tempVisVar(visVar);
        int maxi = -1;
        for(int i : relation[node]) if(i > maxi) maxi = i;
        for(int i = 0; i <= maxi; i++) {
            if(!visVar[i]) {
                visVar[i] = true;
                countRels[i].push_back(node);
            }
        }
        for(int i = 0; i < children[node].size(); i++) initCountRels(children[node][i]);
        visVar = tempVisVar;
        return;
    }

    

    int treeUpp(int node, Bucket &B) {
        vector<int> lower_bound = {};
        vector<int> upper_bound = {};
        for(int j = 0; j < relation[node].size(); j++) {
            lower_bound.push_back(B.getLowerBound()[relation[node][j]]);
            upper_bound.push_back(B.getUpperBound()[relation[node][j]]);
        }
        int sum_cnt = CO[node]->sumCnt(Point<int>(lower_bound), Point<int>(upper_bound));
        if(!sum_cnt) return 0;
        else if(relation[node][relation[node].size() - 1] < B.getSplitDim()) {
            int temp = 1;
            for(int child : children[node]) temp *= treeUpp(child, B);
            return temp;
        }
        else return sum_cnt;
    }

    int treeUpp(int node, int splitDim, vector<pair<vector<int>, vector<int> > > &bound) {
        int sum_cnt = CO[node]->sumCnt(Point<int>(bound[node].first), Point<int>(bound[node].second));
        if(!sum_cnt) return 0;
        else if(relation[node][relation[node].size() - 1] < splitDim) {
            int temp = 1;
            for(int child : children[node]) temp *= treeUpp(child, splitDim, bound);
            return temp;
        }
        else return sum_cnt;
    }

public:
    int root;
    vector<vector<int> > countRels; // countRels[i] = the relations that are needed to count when the splitDim is i

    JoinTree(){}

    JoinTree(Query q, vector<CountOracle<int>*> CO) : CO(CO) {
        queue<int> que;
        root = 0;
        relation = q.getRelations();
        children = vector<vector<int> >(q.getRelNames().size(), vector<int>());
        joinPos = vector<vector<vector<int > > >(q.getRelNames().size(), vector<vector<int> >());
        parent = vector<int>(q.getRelNames().size(), -1);
        countRels = vector<vector<int> >(q.getVarNumber(), vector<int>());
        visVar = vector<bool>(q.getVarNumber(), false);
        cache.resize(q.getRelNames().size());
        treeBound.resize(q.getRelNames().size());
        vector<bool> visited(q.getRelNames().size(), false);
        visited[0] = true;
        que.push(0);
        while(!que.empty()){
            int rel = que.front(); // get the front of the queue
            que.pop();
            vector<int> neighbors = q.getNeighborRels(rel); // get the neighbors of the current relation
            for(int i = 0; i < neighbors.size(); i++) {
                int neighbor = neighbors[i];
                if(visited[neighbor]) continue; // if the neighbor has already been visited, skip it
                
                vector<int> jpos = {};
                for(int j = 0; j < q.getRelations()[rel].size(); j++)
                    if(q.getRelations()[rel][j] == q.getRelations()[neighbor][jpos.size()])
                        jpos.push_back(j);
                
                vector<int> neineighbors = q.getNeighborRels(neighbor, jpos.size());

                bool cyclic = false;
                for(int j = 0; j < neineighbors.size(); j++) {
                    int neineighbor = neineighbors[j];
                    if(neineighbor == rel) continue;
                    if(visited[neineighbor])cyclic = true;
                }

                if(!cyclic) {
                    children[rel].push_back(neighbor);
                    joinPos[rel].push_back(jpos);
                    parent[neighbor] = rel;
                    visited[neighbor] = true;
                    que.push(neighbor); // add the neighbor to the queue for further exploration
                }
            }
        }
        buildLeaves(root);
        for(int i = 0; i < cache.size(); i++) {
            cout << cache[i].size() << " ";
        }
        cout << endl;
            auto startJT = std::chrono::high_resolution_clock::now();
        preProcessing(root);
            auto endJT = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsedJT = endJT - startJT;
            cout << "Time to build the JoinTree: " << elapsedJT.count() << " s\n";
        initCountRels(root);
    }

    int treeUpp(int splitDim, const vector<pair<vector<Point<int> >::iterator, vector<Point<int> >::iterator> > iters) {
        // cout << "TREEUPP IN";
        if(splitDim >= (int)countRels.size()) return 1;
        int tupp = 1;
        for(int node : countRels[splitDim]) {
            tupp *= CO[node]->sumCnt(iters[node].first, iters[node].second);
        }
        // cout << "TREEUPP OUT" << endl;
        return tupp;
    }

    int treeUpp(int splitDim, vector<pair<vector<int>, vector<int> > > &bound) {
        return treeUpp(root, splitDim, bound);
    }
    
    int treeUpp(Bucket &B) {
        return treeUpp(root, B);
    }

    void printTree(int nodeID, int depth = 0) {
        for (int i = 0; i < depth; i++) {
            cout << "| ";
        }
        cout << "R" << nodeID << endl;

        for (int childID : children[nodeID]) {
            printTree(childID, depth + 1);
        }

    }

    void printChildren() {
        for(int i = 0; i < children.size(); i++) {
            cout << "R" << i << " has children: ";
            for(int j = 0; j < children[i].size(); j++) {
                cout << "R" << children[i][j] << "(";
                for(int k = 0; k < joinPos[i][j].size() - 1; k++) {
                    cout << joinPos[i][j][k] << ", ";
                }
                cout << joinPos[i][j][joinPos[i][j].size() - 1] << "); ";
            }
            cout << endl;
        }
    }

    void print(){
        printTree(root);
    }
};