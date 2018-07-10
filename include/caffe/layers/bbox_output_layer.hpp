#ifndef CAFFE_BBOX_OUTPUT_LAYER_HPP_
#define CAFFE_BBOX_OUTPUT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class BBoxOutputLayer : public Layer<Dtype> {
 public:
  explicit BBoxOutputLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);
  virtual void Reshape(
	  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "BBoxOutput"; }

  virtual inline int ExactNumBottomBlobs() const { return 4; }
  virtual inline int ExactNumTopBlobs() const { return 0; }
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return false;
  }

 protected:
  /// @copydoc BBoxOutputLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
	  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  vector<std::string> lines_;
  int lines_id_;
};

}  // namespace caffe

#endif  // CAFFE_BBOX_OUTPUT_LAYER_HPP_
