#include "BanPickTree.hpp"
int main(){
    BanPickTree bp(10);
    bp.ban(2,4);
    while(bp.remaining() > 0){
        cout << "pick: " << bp.pick() << endl;
        if(bp.available(6,8))bp.ban(6,8);
        bp.print();
    }

    return 0;
}