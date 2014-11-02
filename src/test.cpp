#include "constant.h"
#include "stdio.h"

#include "SgdLogReg.h"

int main()
{
    printf("start sgd for lr[%u]\n", shishu::sgd::MAX_INPUT_LENGTH);
    char conf_file[20] = "./sgd_conf";
    shishu::sgd::SgdLogReg slr;
    slr.load_conf(conf_file);
    slr.load_train_data();
    slr.load_test_data();
    slr.sgd_calc_model();
    slr.predict_test_data();
    return 0;
}
