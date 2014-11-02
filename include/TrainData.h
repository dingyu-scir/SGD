// @file include/TrainData.h
// @author shishu
// @date 2014/06/13
// @brief TrainData used to save one sample of training data

#ifndef SHISHU_SGD_TRAINDATA_H
#define SHISHU_SGD_TRAINDATA_H

#include "stdio.h"
#include "Fea.h"

namespace shishu {
namespace sgd {

// @brief one line of train data ,store features by vector, also save the tag value
class TrainData {
public:
    TrainData(uint32_t size)
    {
        if (size == 0) {
            printf("size is zero!\n");
        }

        _fea_num = size;
        _fea_list = new Fea[size];
        _cur_fea_id = 0;
        _label = -1;
    }

    int add_fea(uint32_t id, double val)
    {
        if (_cur_fea_id >= _fea_num) {
            printf("add fea out of band, id[%u], val[%lf], _fea_num[%u]", id, val, _fea_num);
            return -1;
        }

        _fea_list[_cur_fea_id].id = id;
        _fea_list[_cur_fea_id++].val = val;
        return 0;
    }

    void set_label(int label)
    {
        _label = (1 + label) / 2;
    }

    int label()
    {
        return _label;
    }

private:
    TrainData()
    {
        // do nothing
    }

public:
    uint32_t _fea_num;
    uint32_t _cur_fea_id;
    Fea* _fea_list;
    int _label;
};

}; // @namespace sgd
}; // @namespace shishu

#endif
