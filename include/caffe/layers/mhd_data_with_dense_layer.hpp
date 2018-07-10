#ifndef CAFFE_MHD_DATA_WITH_DENSE_LAYER_HPP_
#define CAFFE_MHD_DATA_WITH_DENSE_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include <itkImage.h>
#include <itkSmartPointer.h>
#include <itkImageFileReader.h>
#include <itkImageFileWriter.h>

namespace caffe {

/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class MHDDataWithDenseLayer : public ExBasePrefetchingDataLayer<Dtype> {
 public:
  typedef itk::Image<Dtype, 3> ImageType;
  typedef itk::Image<char, 3> LabelType;

  explicit MHDDataWithDenseLayer(const LayerParameter& param)
      : ExBasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~MHDDataWithDenseLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "MultiChannelData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 3; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void load_batch(ExBatch<Dtype>* batch);
  shared_ptr<Caffe::RNG> ind_rng_;
  shared_ptr<Caffe::RNG> trans_rng_;
  vector<std::pair<std::string, std::pair<itk::SmartPointer<ImageType>, itk::SmartPointer<LabelType>>>> lines_;
  int lines_id_;
};

}  // namespace caffe

#endif  // CAFFE_MHD_DATA_WITH_DENSE_LAYER_HPP_
