#include "SgdLogReg.h"
#include <string>
#include <math.h>

namespace shishu {
namespace sgd {

int SgdLogReg::load_conf(char* conf_file)
{
    _fea_num = 125;
    _record_num = 25000;
    _sample_path = "./train_data";
    _output_path = "./sgd_model";
    _converge_rate = 0.0001;
    _ds = new DataSet(_record_num);
    _iter_num = 10000;
    _wv = new double[_fea_num];
    for (uint32_t i = 0; i < _fea_num; i++) {
        _wv[i] = 0;
    }
    return 0;
}

int SgdLogReg::load_data()
{
    FILE* file = fopen(_sample_path, "r");
    if (file == NULL) {
        printf("can't open file [%s]", _sample_path);
        return -1;
    }

    char input[MAX_INPUT_LENGTH] = {'\0'};
    while (fgets(input, MAX_INPUT_LENGTH, file) != NULL) {
        input[strlen(input) - 1] = '\0';
        TrainData** td =  _ds->get_one_data();
        uint32_t fea_num = 0;
        for (uint32_t i = 0; i < strlen(input); i++) {
            if (input[i] == ':') {
                fea_num ++;
            }
        }
        *td = new TrainData(fea_num + 1);
        printf("train data has [%u] features \n", fea_num + 1);
        char *str = NULL;
        str = strtok(input, SEG_TOKEN);

        int tag = atoi(str);
        (*td)->set_label(tag);
        printf("train data tag is [%d]\n", tag);
        for (uint32_t j = 0; j < fea_num; j++) {
            str = strtok(NULL, SEG_TOKEN);
            if (str == NULL) {
                printf("load data error!!");
                return -1;
            }
            uint32_t idx = 0;
            double value = 0.0f;
            if (get_id_and_value(str, &idx, &value) != 0) {
                printf("analyze feature_field failed!!");
                return -1;
            }
            (*td)->add_fea(idx, value);
        }
        (*td)->add_fea(0, 1);
        printf("\n");
    }

    return 0;
}

int SgdLogReg::get_id_and_value(char* str, uint32_t* idx, double* value)
{
    uint32_t str_len = strlen(str);
    char idx_str[MAX_INPUT_LENGTH] = {'\0'};
    char value_str[MAX_INPUT_LENGTH] = {'\0'};
    int tag = -1;

    for (uint32_t i = 0; i < str_len; i++) {
        if (str[i] == SEG_COLON) {
            tag = i;
            idx_str[i] = '\0';
        }

        if (tag < 0) {
            idx_str[i] = str[i];
        } else if (static_cast<int>(i) > tag) {
            value_str[i - tag - 1] = str[i];
        }
    }
    value_str[str_len - 1 - tag] = '\0';

    *idx = static_cast<uint32_t>(atoi(idx_str));
    *value =atof(value_str);
    printf("\tid[%u]:val[%lf]", *idx, *value);
    return 0;
}

double SgdLogReg::calc_predict_res(TrainData* td, double* wv)
{
    double total = 0;
    for (uint32_t i = 0; i < td->_fea_num; i++) {
        uint32_t idx = td->_fea_list[i].id;
        double val = td->_fea_list[i].val;
        total += val * wv[idx];
    }
    total = 0 - total;
    printf("total is [%lf]", total);
    return (1/(1 + exp(total)));
}

int SgdLogReg::sgd_calc_model()
{
    double cur_rate = 1.0;
    double last_error = 0;
    double cur_iter = 1;
    while (cur_rate >= _converge_rate) {
        double step_length = 1 / (cur_iter);
        double cur_error = 0.0f;
        for (uint32_t i = 0; i < _iter_num; i++) {
            TrainData* td = _ds->get_rand_data();
            double p_res = calc_predict_res(td, _wv);
            printf("p_res is [%lf]\n", p_res);
            printf("label is [%d]\n", td->_label);
            for (uint32_t j = 0; j < td->_fea_num; j++) {
                uint32_t idx = td->_fea_list[j].id;
                _wv[idx] = _wv[idx] + step_length * (td->_label - p_res);
                printf("idx[%u], new w[%lf]\n", idx, _wv[idx]);
            }
            cur_error += (td->_label - p_res) * (td->_label - p_res);
        }

        if (cur_error > last_error) {
            cur_rate = 1.0;
        } else {
            cur_rate = (last_error - cur_error) / last_error;
        }
        
        printf("cur_iter[%lf] cur_error[%lf] cur_rate[%lf] step_length[%lf]\n",
                cur_iter, cur_error, cur_rate, step_length);
        cur_iter++;

        last_error = cur_error;

        if (cur_iter == 20)
            break;
    }
    return 0;
}

};
};
