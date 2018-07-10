#ifndef CAFFE_SLICE_PREDICT_LAYER_HPP_
#define CAFFE_SLICE_PREDICT_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include <itkImage.h>
#include <itkSmartPointer.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

namespace caffe {

template <typename Dtype>
class SlicePredictLayer : public Layer<Dtype> {
 public:
  typedef itk::Image<Dtype, 3> ImageType;
  typedef itk::Image<char, 3> LabelType;
  typedef itk::Image<Dtype, 2> ImageSliceType;
  typedef itk::Image<char, 2> LabelSliceType;

  explicit SlicePredictLayer(const LayerParameter& param)
	  : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SlicePredict"; }

  virtual inline int MinBottomBlobs() const { return 1; }
  virtual inline int MaxBottomBlobs() const { return 2; }
  virtual inline int MinTopBlobs() const { return 0; }
  virtual inline int MaxTopBlobs() const { return 2; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

 protected:
  float* anchors_;
  int num_anchors_;
  int feat_size_[3];
  vector<std::pair<std::string, std::pair<int, int>>> lines_;
  int lines_id_;
  shared_ptr<Caffe::RNG> rng_;
};


}  // namespace caffe

#endif  // CAFFE_SLICE_PREDICT_LAYER_HPP_
