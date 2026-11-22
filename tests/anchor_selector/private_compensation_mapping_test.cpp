#define CARTGS_TEST_PRIVATE_COMPENSATION
#define main variable_anchor_lifecycle_main
#include "variable_anchor_test.cpp"
#undef main

int main(int argc,char** argv) { return variable_anchor_lifecycle_main(argc,argv); }
