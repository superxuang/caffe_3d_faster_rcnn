#ifndef CAFFE_SEGMENT_OUTPUT_LAYER_HPP_
#define CAFFE_SEGMENT_OUTPUT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include <itkImage.h>
#include <itkSmartPointer.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>
#include <itkIdentityTransform.h>
#include <itkResampleImageFilter.h>
#include <itkNearestNeighborInterpolateImageFunction.h>

namespace caffe {

template <typename Dtype>
class SegmentOutputLayer : public Layer<Dtype> {
 public:
  typedef itk::Image<Dtype, 3> ImageType;
  typedef itk::Image<char, 3> LabelType;
  typedef itk::ImageFileReader<ImageType> ImageReaderType;
  typedef itk::ImageFileReader<LabelType> LabelReaderType;
  typedef itk::ImageFileWriter<ImageType> ImageWriterType;
  typedef itk::ImageFileWriter<LabelType> LabelWriterType;
  typedef itk::IdentityTransform<double, 3> IdentityTransformType;
  typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleImageFilterType;
  typedef itk::ResampleImageFilter<LabelType, LabelType> ResampleLabelFilterType;
  typedef itk::NearestNeighborInterpolateImageFunction<LabelType, double> InterpolatorType;

  explicit SegmentOutputLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	  const vector<Blob<Dtype>*>& top);
  virtual void Reshape(
	  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return bottom_index != 1;
  }

  virtual inline const char* type() const { return "SegmentOutput"; }

 protected:
  /// @copydoc SegmentOutputLayer
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
	  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  vector<std::string> lines_;
  int lines_id_;
};

}  // namespace caffe

#endif  // CAFFE_SEGMENT_OUTPUT_LAYER_HPP_
