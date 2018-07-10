#include <algorithm>
#include <vector>

#include "caffe/layers/segment_output_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include <itkMetaImageIOFactory.h>
#include <itkConnectedComponentImageFilter.h>
#include <itkContourDirectedMeanDistanceImageFilter.h>
#include <itkDirectedHausdorffDistanceImageFilter.h>

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
void SegmentOutputLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const SegmentOutputParameter& segment_output_param = this->layer_param_.segment_output_param();
  const ContourNameList& contour_name_list = segment_output_param.contour_name_list();
  const int contour_num = contour_name_list.name_size();
  const string& source = segment_output_param.source();
  const string& root_folder = segment_output_param.root_folder();
  itk::ObjectFactoryBase::RegisterFactory(itk::MetaImageIOFactory::New());

  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string line;
  size_t pos1, pos2;
  while (std::getline(infile, line)) {
    pos1 = line.find_first_of(' ');
    pos2 = line.find_last_of(' ');
    string image_file_name = line.substr(0, pos1);
    string label_file_name = line.substr(pos1 + 1, pos2 - pos1 - 1);
    string info_file_name = line.substr(pos2 + 1);
    lines_.push_back(image_file_name);
  }

  CHECK(!lines_.empty()) << "File is empty";

  lines_id_ = 0;
}

template <typename Dtype>
void SegmentOutputLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape(0);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void SegmentOutputLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* score_data = bottom[0]->cpu_data();
  const Dtype* image_info = bottom[1]->cpu_data();

  const int num = bottom[0]->num();
  const int channel = bottom[0]->channels();
  const int count = bottom[0]->count();
  const int dim = count / num;
  const int pixel_num = dim / channel;
  const SegmentOutputParameter& segment_output_param = this->layer_param_.segment_output_param();
  const ContourNameList& contour_name_list = segment_output_param.contour_name_list();
  const int contour_num = contour_name_list.name_size();
  const string& root_folder = segment_output_param.root_folder();
  const string& output_folder = segment_output_param.output_folder();

  Dtype* loss = top[0]->mutable_cpu_data();
  loss[0] = 0;

  typedef itk::IdentityTransform<double, 3> IdentityTransformType;
  typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleImageFilterType;
  ImageType::SizeType size_btm, size_out;
  ImageType::SpacingType spacing_btm, spacing_out;
  ImageType::PointType origin_btm, origin_out;
  ImageType::DirectionType direct_btm;
  ImageType::IndexType index_btm;
  ImageType::RegionType region_btm;

  for (int i = 0; i < num; ++i) {
    size_t pos = lines_[lines_id_].find_first_of('/');
    std::string file_name = lines_[lines_id_].substr(pos + 1);
    ImageType::SizeType size_src;
    ImageType::RegionType region_src;
    ImageType::SpacingType spacing_src;
    ImageType::PointType origin_src;
	ImageType::DirectionType direct_src;
	for (int j = 0; j < 3; ++j) {
      size_src[j] = image_info[i * 18 + 9 + j];
      spacing_src[j] = image_info[i * 18 + 12 + j];
      origin_src[j] = image_info[i * 18 + 15 + j];
      for (int k = 0; k < 3; ++k) {
        direct_src[j][k] = (j == k) ? 1.0 : 0.0;
      }
	}
	region_src.SetSize(size_src);
	CHECK_EQ(region_src.GetNumberOfPixels(), pixel_num);

    ImageType::SizeType size_resample;
	ImageType::RegionType region_resample;
	ImageType::SpacingType spacing_resample;
	ImageType::PointType origin_resample;
	ImageType::DirectionType direct_resample = direct_src;
	for (int k = 0; k < 3; ++k) {
      size_resample[k] = image_info[i * 18 + 0 + k];
      spacing_resample[k] = image_info[i * 18 + 3 + k];
	  origin_resample[k] = image_info[i * 18 + 6 + k];
	}
    region_resample.SetSize(size_resample);

	LabelType::Pointer mask = LabelType::New();
	mask->SetDirection(direct_src);
	mask->SetOrigin(origin_src);
	mask->SetSpacing(spacing_src);
	mask->SetRegions(region_src);
	mask->Allocate();
	char* mask_buffer = mask->GetBufferPointer();
	memset(mask_buffer, 0, sizeof(char) * pixel_num);

	Dtype* mask_max_score = new Dtype[pixel_num];
	memset(mask_max_score, 0, sizeof(Dtype) * pixel_num);

    for (int j = 0; j < contour_num + 1; ++j) {
      ImageType::Pointer score_map = ImageType::New();
      score_map->SetDirection(direct_src);
      score_map->SetOrigin(origin_src);
      score_map->SetSpacing(spacing_src);
      score_map->SetRegions(region_src);
      score_map->Allocate();
      Dtype* score_map_buffer = score_map->GetBufferPointer();
	  memcpy(score_map_buffer, score_data + i * dim + j * pixel_num, sizeof(Dtype) * pixel_num);
	  for (int k = 0; k < pixel_num; ++k) {
        double score_value = score_map_buffer[k];
        if (mask_max_score[k] < score_value) {
          mask_buffer[k] = j;
          mask_max_score[k] = score_value;
        }
      }
      if (j > 0 && segment_output_param.output_heatmap()) {
        // resample to origin size  
        ResampleImageFilterType::Pointer score_map_resampler = ResampleImageFilterType::New();
		score_map_resampler->SetInput(score_map);
		score_map_resampler->SetSize(size_resample);
		score_map_resampler->SetOutputSpacing(spacing_resample);
		score_map_resampler->SetOutputOrigin(origin_resample);
        score_map_resampler->SetTransform(IdentityTransformType::New());
        score_map_resampler->Update();
		score_map = score_map_resampler->GetOutput();
		score_map->SetDirection(direct_resample);
  
		const string score_map_filename = output_folder + "score_" + contour_name_list.name(j - 1) + "_" + file_name;
        ImageWriterType::Pointer score_map_writer = ImageWriterType::New();
		score_map_writer->SetInput(score_map);
        score_map_writer->SetFileName(score_map_filename);
        score_map_writer->Update();
      }
    }
	delete[]mask_max_score;
	ResampleLabelFilterType::Pointer mask_resampler = ResampleLabelFilterType::New();
	mask_resampler->SetInput(mask);
	mask_resampler->SetSize(size_resample);
	mask_resampler->SetOutputSpacing(spacing_resample);
	mask_resampler->SetOutputOrigin(origin_resample);
	mask_resampler->SetTransform(IdentityTransformType::New());
	mask_resampler->SetInterpolator(InterpolatorType::New());
	mask_resampler->Update();
	mask = mask_resampler->GetOutput();
	mask->SetDirection(direct_resample);

	const string mask_filename = output_folder + "mask_" + file_name;
	LabelWriterType::Pointer mask_writer = LabelWriterType::New();
	mask_writer->SetInput(mask);
	mask_writer->SetFileName(mask_filename);
	mask_writer->Update();

    lines_id_ = (lines_id_ + 1) % lines_.size();
  }
}

template <typename Dtype>
void SegmentOutputLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	  const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

INSTANTIATE_CLASS(SegmentOutputLayer);
REGISTER_LAYER_CLASS(SegmentOutput);

}  // namespace caffe
