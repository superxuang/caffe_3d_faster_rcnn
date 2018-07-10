#include <algorithm>
#include <cfloat>
#include <vector>
#include <iostream>

#include "caffe/layers/roi_index_layer.hpp"

namespace caffe {

template <typename Dtype>
void RoiIndexLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void RoiIndexLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	num_ = bottom[0].shape(0);
	channels_ = bottom[0].shape(1);
	length_ = bottom[0].shape(2);
	height_ = bottom[0].shape(3);
	width_ = bottom[0].shape(4);
	roi_num_ = bottom[1].shape(0);
	top[0].Reshape(roi_num_, channels_, length_, height_, width_);
}

template <typename Dtype>
void RoiIndexLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	const int grid_size = 8;
	const int bin_size = width_ / grid_size;
	const int bin_size_z = length_ / grid_size;
	const Dtype* ind_data = bottom[0].cpu_data();
	const Dtype* roi_data = bottom[1].cpu_data();
	for (int roi_id = 0; roi_id < roi_num_; ++roi_id) {
		const int roi_x_start = roi_data[roi_id * 7 + 0];
		const int roi_x_end = roi_data[roi_id * 7 + 1];
		const int roi_y_start = roi_data[roi_id * 7 + 2];
		const int roi_y_end = roi_data[roi_id * 7 + 3];
		const int roi_z_start = roi_data[roi_id * 7 + 4];
		const int roi_z_end = roi_data[roi_id * 7 + 5];
		const int roi_scale = roi_data[roi_id * 7 + 6];
		for (int z = roi_z_start * bin_size_z; z < roi_z_end * bin_size_z; ++z) {
			for (int y = roi_y_start * bin_size; y < roi_y_end * bin_size; ++y) {
				for (int x = roi_x_start * bin_size; x < roi_x_end * bin_size; ++x) {
					ind_data[z * height_ * width_ + y * width_ + x]
				}
			}
		}
	}
}

template <typename Dtype>
void RoiIndexLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}


#ifdef CPU_ONLY
STUB_GPU(RoiIndexLayer);
#endif

INSTANTIATE_CLASS(RoiIndexLayer);
REGISTER_LAYER_CLASS(RoiIndex);

}  // namespace caffe
