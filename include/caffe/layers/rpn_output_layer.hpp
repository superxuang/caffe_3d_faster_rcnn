#ifndef CAFFE_RPN_OUTPUT_LAYER_HPP_
#define CAFFE_RPN_OUTPUT_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/fast_rcnn_layers.hpp"

namespace caffe {

template <typename Dtype>
class RPNOutputLayer : public Layer<Dtype> {
 public:
  explicit RPNOutputLayer(const LayerParameter& param)
	  : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "RPNOutput"; }

  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 0; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
	  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 protected:
  double* anchors_;
  int anchors_num_;
  int feat_size_[3];
  vector<std::pair<std::string, std::pair<std::string, std::string>>> lines_;
  int lines_id_;
  shared_ptr<Caffe::RNG> rng_;
};


}  // namespace caffe

#endif  // CAFFE_RPN_OUTPUT_LAYER_HPP_
