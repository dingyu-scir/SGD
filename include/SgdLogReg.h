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
        : _fea_num(0)
        , _record_num(0)
        , _sample_path(NULL)
        , _output_path(NULL)
        , _converge_rate(0.01)
        , _ds(NULL)
        , _iter_num(0)
        , _wv(NULL)
    {
        // do nothing
    }

    ~SgdLogReg()
    {
        // do nothing
    }

    int load_conf(char* conf_file);

    int load_data();

    int sgd_calc_model();

private:
    double calc_predict_res(TrainData* td, double* wv);

    int get_id_and_value(char* str, uint32_t* idx, double* value);

private:
    uint32_t _fea_num;
    uint32_t _record_num;
    char* _sample_path;
    char* _output_path;
    double _converge_rate;
    DataSet* _ds;
    uint32_t _iter_num;
    double* _wv;
};

}; // @namespace sgd
}; // @namespace shishu

#endif
