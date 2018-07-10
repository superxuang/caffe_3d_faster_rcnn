#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/roi_index_layer.hpp"

namespace caffe {

template <typename Dtype>
void RoiIndexLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void RoiIndexLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}


INSTANTIATE_LAYER_GPU_FUNCS(RoiIndexLayer);


}  // namespace caffe
