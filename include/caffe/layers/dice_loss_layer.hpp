#ifndef CAFFE_DICE_LOSS_LAYER_HPP_
#define CAFFE_DICE_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class DiceLossLayer : public LossLayer<Dtype> {
 public:
  explicit DiceLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {
	class_exist_ = NULL;
  }
  virtual ~DiceLossLayer() {
	delete[]class_exist_;
  }

  virtual inline const char* type() const { return "DiceLoss"; }
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(
	  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

 protected:
  /// @copydoc DiceLossLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  //virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	 // const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  //virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
	 // const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> intersection_;
  Blob<Dtype> union_;
  bool* class_exist_;
  Blob<Dtype> buffer_;
  Blob<Dtype> ones_mask_;
};

}  // namespace caffe

#endif  // CAFFE_DICE_LOSS_LAYER_HPP_
