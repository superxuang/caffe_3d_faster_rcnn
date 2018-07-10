#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/rpn_output_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

using std::sqrt;
using std::floor;
using std::ceil;
using std::min;
using std::max;
using std::exp;
using std::log;

namespace caffe {

template <typename Dtype>
void RPNOutputLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

	const unsigned int rng_seed = caffe_rng_rand();
	rng_.reset(new Caffe::RNG(rng_seed));

	const RPNOutputParameter& rpn_output_param = this->layer_param_.rpn_output_param();

	const AnchorParameter& anchor_param = rpn_output_param.anchor_size();
	anchors_ = caffe::GenerateAnchors(anchor_param, anchors_num_);

	const string& source = rpn_output_param.source();
	std::ifstream infile(source.c_str());
	string line;
	size_t pos1, pos2;
	while (std::getline(infile, line)) {
		pos1 = line.find_first_of(' ');
		pos2 = line.find_last_of(' ');
		lines_.push_back(std::make_pair(line.substr(0, pos1),
		std::make_pair(line.substr(pos1 + 1, pos2 - pos1 - 1), line.substr(pos2 + 1))));
	}

	lines_id_ = 0;
}

template <typename Dtype>
void RPNOutputLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
  // Reshaping happens during the call to forward.
}

template <typename Dtype>
void RPNOutputLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void RPNOutputLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

INSTANTIATE_CLASS(RPNOutputLayer);
REGISTER_LAYER_CLASS(RPNOutput);

}  // namespace caffe