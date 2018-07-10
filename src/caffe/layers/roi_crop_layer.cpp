// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include <cfloat>

#include "caffe/layers/roi_crop_layer.hpp"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe {

template <typename Dtype>
void ROICropLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ROICropParameter roi_crop_param = this->layer_param_.roi_crop_param();
  spatial_scale_xy_ = roi_crop_param.spatial_scale_xy();
  spatial_scale_z_ = roi_crop_param.spatial_scale_z();
}

template <typename Dtype>
void ROICropLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  length_ = bottom[0]->shape(2);
  height_ = bottom[0]->shape(3);
  width_ = bottom[0]->shape(4);
  Dtype* roi_data = bottom[1]->mutable_cpu_data();
  crop_start_w_ = round(roi_data[1] * spatial_scale_xy_);
  crop_start_h_ = round(roi_data[2] * spatial_scale_xy_);
  crop_start_l_ = round(roi_data[3] * spatial_scale_z_);
  int crop_end_w = round(roi_data[4] * spatial_scale_xy_);
  int crop_end_h = round(roi_data[5] * spatial_scale_xy_);
  int crop_end_l = round(roi_data[6] * spatial_scale_z_);
  crop_length_ = max(crop_end_l - crop_start_l_ + 1, 1);
  crop_height_ = max(crop_end_h - crop_start_h_ + 1, 2);
  crop_width_ = max(crop_end_w - crop_start_w_ + 1, 2);
  std::vector<int> top_shape(5);
  top_shape[0] = num_;
  top_shape[1] = channels_;
  top_shape[2] = crop_length_;
  top_shape[3] = crop_height_;
  top_shape[4] = crop_width_;
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void ROICropLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  Dtype* bottom_data = bottom[0]->mutable_cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int top_count = top[0]->count();
  caffe_set(top_count, Dtype(0), top_data);

  for (int n = 0; n < num_; ++n) {
    for (int c = 0; c < channels_; ++c) {
      std::vector<int> offset_inds(5);
      offset_inds[0] = n;
      offset_inds[1] = c;
      offset_inds[2] = 0;
      offset_inds[3] = 0;
      offset_inds[4] = 0;      
      bottom_data = bottom[0]->mutable_cpu_data() + bottom[0]->offset(offset_inds);
      top_data = top[0]->mutable_cpu_data() + top[0]->offset(offset_inds);
      for (int top_l = 0; top_l < crop_length_; ++top_l) {
        for (int top_h = 0; top_h < crop_height_; ++top_h) {
          for (int top_w = 0; top_w < crop_width_; ++top_w) {
        
			int bottom_l = top_l + crop_start_l_;
			int bottom_h = top_h + crop_start_h_;
			int bottom_w = top_w + crop_start_w_;

			if (bottom_l >= 0 && bottom_l < length_ &&
				bottom_h >= 0 && bottom_h < height_ &&
				bottom_w >= 0 && bottom_w < width_) {
			  const int bottom_index = bottom_l * height_ * width_ + bottom_h * width_ + bottom_w;
			  const int top_index = top_l * crop_height_ * crop_width_ + top_h * crop_width_ + top_w;
			  top_data[top_index] = bottom_data[bottom_index];
			}        
          }
        }
      }
    }
  }
}

template <typename Dtype>
void ROICropLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  Dtype* top_diff = top[0]->mutable_cpu_diff();
  int bottom_count = bottom[0]->count();
  caffe_set(bottom_count, Dtype(0), bottom_diff);

  for (int n = 0; n < num_; ++n) {
    for (int c = 0; c < channels_; ++c) {
      std::vector<int> offset_inds(5);
      offset_inds[0] = n;
      offset_inds[1] = c;
      offset_inds[2] = 0;
      offset_inds[3] = 0;
      offset_inds[4] = 0;      
      bottom_diff = bottom[0]->mutable_cpu_diff() + bottom[0]->offset(offset_inds);
      top_diff = top[0]->mutable_cpu_diff() + top[0]->offset(offset_inds);
      for (int top_l = 0; top_l < crop_length_; ++top_l) {
        for (int top_h = 0; top_h < crop_height_; ++top_h) {
          for (int top_w = 0; top_w < crop_width_; ++top_w) {
        
			int bottom_l = top_l + crop_start_l_;
			int bottom_h = top_h + crop_start_h_;
			int bottom_w = top_w + crop_start_w_;

			if (bottom_l >= 0 && bottom_l < length_ &&
				bottom_h >= 0 && bottom_h < height_ &&
				bottom_w >= 0 && bottom_w < width_) {
			  const int bottom_index = bottom_l * height_ * width_ + bottom_h * width_ + bottom_w;
			  const int top_index = top_l * crop_height_ * crop_width_ + top_h * crop_width_ + top_w;
			  bottom_diff[bottom_index] = top_diff[top_index];
			}        
          }
        }
      }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(ROICropLayer);
#endif

INSTANTIATE_CLASS(ROICropLayer);
REGISTER_LAYER_CLASS(ROICrop);

}  // namespace caffe
