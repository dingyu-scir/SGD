// @file include/DataSet.h
// @author shishu
// @date 2014/06/13
// @brief DataSet used to store train datas

#ifndef SHISHU_SGD_DATASET_H
#define SHISHU_SGD_DATASET_H

#include "constant.h"
#include "stdint.h"
#include "stdio.h"
#include "stdlib.h"
#include "time.h"
#include "TrainData.h"

namespace shishu {
namespace sgd {

// @brief data set used to store train datas
// each train data is stored as a vary-length vector whiched named TrainData
class DataSet {
public:
    DataSet(uint32_t data_size)
    {
        if (data_size == 0) {
            printf("size is zero");
        }

        _data_num = data_size;
        _data_list = new TrainData*[data_size];
        _cur_data_id = 0;
        srand((int)time(NULL));
    }

    TrainData** get_one_data()
    {
        if (_cur_data_id >= _data_num) {
            printf("no other available slot");
            return static_cast<TrainData**>(NULL);
        }

        _cur_data_id++;

        return &_data_list[_cur_data_id - 1];
    }

    TrainData* get_rand_data()
    {
        uint32_t idx = (rand() % _data_num);
        printf("rand idx [%u]\n", idx);
        return _data_list[idx];
    }

    TrainData* get_idx_data(uint32_t idx)
    {
        if (idx < _cur_data_id) {
            return _data_list[idx];
        }
        
        return NULL;
    }

    uint32_t length()
    {
        return _cur_data_id;
    }

private:
    DataSet()
    {
        // do nothing
    }

private:
    uint32_t _data_num;
    uint32_t _cur_data_id;
    TrainData** _data_list;
};

}; // @namespace sgd
}; // @namespace shishu

#endif
