#include <fstream>
#include <iostream>
int main(int argc,char** argv) {
    if(argc!=2)return 2;
    std::ifstream file(argv[1],std::ios::binary);
    std::cout<<"STDLIB_STREAM_OPEN "<<file.is_open()<<std::endl;
    return file?0:1;
}
