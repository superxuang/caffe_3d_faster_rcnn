#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/proposal_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
void ProposalLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

	const unsigned int rng_seed = caffe_rng_rand();
	rng_.reset(new Caffe::RNG(rng_seed));

	const ProposalParameter& proposal_param = this->layer_param_.proposal_param();
	const int proposal_num = proposal_param.proposal_num();

	const AnchorParameter& anchor_param = proposal_param.anchor_size();
	anchors_ = caffe::GenerateAnchors(anchor_param, anchors_num_);

	vector<int> top_shape(2);
	top_shape[0] = 1;
	top_shape[1] = 7;
	top[0]->Reshape(top_shape);
}

template <typename Dtype>
void ProposalLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
  // Reshaping happens during the call to forward.
}

template <typename Dtype>
void ProposalLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void ProposalLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

INSTANTIATE_CLASS(ProposalLayer);
REGISTER_LAYER_CLASS(Proposal);

}  // namespace caffe