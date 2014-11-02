// @author shishu
// @date 2014/06/13
// @brief sgd method for logistic regression

#ifndef SHISHU_SGD_SGDLOGREG_H
#define SHISHU_SGD_SGDLOGREG_H

#include "DataSet.h"

namespace shishu {
namespace sgd {

class SgdLogReg {
public:
    SgdLogReg()
        : _record_num(0)
        , _test_num(0)
        , _fea_num(0)
        , _sample_path(NULL)
        , _test_path(NULL)
        , _output_path(NULL)
        , _converge_rate(0.01)
        , _iter_num(0)
        , _wv(NULL)
        , _train_ds(NULL)
        , _test_ds(NULL)
    {
        // do nothing
    }

    ~SgdLogReg()
    {
        // do nothing
    }

    int load_train_data();
    
    int load_test_data();

    int load_conf(char* conf_file);

    int sgd_calc_model();

    int predict_test_data();

private:
    int predict_ds(DataSet* ds, double* wv);

    double calc_predict_res(TrainData* td, double* wv);

    int get_id_and_value(char* str, uint32_t* idx, double* value);

    // @brief although ds is const, its constness means the pointer value
    // will not change
    int load_data(const char* data_path, DataSet* ds);

private:
    uint32_t _record_num;
    uint32_t _test_num;

    uint32_t _fea_num;
    char* _sample_path;
    char* _test_path;
    char* _output_path;

    double _converge_rate;
    uint32_t _iter_num;
    double* _wv;

    // train_data
    DataSet* _train_ds;
    // test_data
    DataSet* _test_ds;
};

}; // @namespace sgd
}; // @namespace shishu

#endif
