#ifndef CAFFE_ANCHOR_TARGET_LAYER_HPP_
#define CAFFE_ANCHOR_TARGET_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/fast_rcnn_layers.hpp"

namespace caffe {

template <typename Dtype>
class AnchorTargetLayer : public Layer<Dtype> {
 public:
  explicit AnchorTargetLayer(const LayerParameter& param)
	  : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "AnchorTarget"; }

  virtual inline int ExactNumBottomBlobs() const { return 4; }
  virtual inline int ExactNumTopBlobs() const { return 4; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 protected:
  double* anchors_;
  int anchors_num_;
  int feat_size_[3];
  shared_ptr<Caffe::RNG> rng_;

  double bbox_target_num_;
  double bbox_target_mean_[6];
  double bbox_target_mean_2_[6];
};


}  // namespace caffe

#endif  // CAFFE_ANCHOR_TARGET_LAYER_HPP_
