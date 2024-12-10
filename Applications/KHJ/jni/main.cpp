// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 DongHak Park <donghak.park@samsung.com>
 *
 * @file   main.cpp
 * @date   26 Jan 2023
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug	   No known bugs except for NYI items
 * @brief  This is Layer Example with ini file
 *
 */

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <vector>

#include <cifar_dataloader.h>
#include <layer.h>
#include <model.h>
#include <optimizer.h>

using LayerHandle = std::shared_ptr<ml::train::Layer>;
using ModelHandle = std::unique_ptr<ml::train::Model>;
using UserDataType = std::unique_ptr<nntrainer::util::DataLoader>;

unsigned int DATA_SIZE;
unsigned int BATCH_SIZE;
unsigned int INPUT_SHAPE[3];
unsigned int OUTPUT_SHAPE[3];
unsigned int seed;

int main(int argc, char *argv[]) {
  int status = 0;
  seed = time(NULL);
  srand(seed);

  if (argc < 2) {
    std::cout << "Usage: " << argv[0] << " <model_config>" << std::endl;
    return -1;
  }

  auto config = argv[1];

  std::unique_ptr<ml::train::Model> model;

  try {
    model = createModel(ml::train::ModelType::NEURAL_NET);
  } catch (std::exception &e) {
    std::cerr << "Error while creating model! details: " << e.what()
              << std::endl;
    return 1;
  }

  try {
    model->load(config, ml::train::ModelFormat::MODEL_FORMAT_INI);
  } catch (std::exception &e) {
    std::cerr << "Error while loading model! details: " << e.what()
              << std::endl;
    return 1;
  }

  try {
    model->compile();
  } catch (std::exception &e) {
    std::cerr << "Error while compiling model! details: " << e.what()
              << std::endl;
    return 1;
  }

  try {
    model->initialize();
  } catch (std::exception &e) {
    std::cerr << "Error while initializing model! details: " << e.what()
              << std::endl;
    return 1;
  }

  auto input_dim = model->getInputDimension();
  auto output_dim = model->getOutputDimension();

  for (auto &dim : input_dim) {
    std::cout << "INPUT_SHAPE : " << dim.channel() << ", " << dim.height()
              << ", " << dim.width() << std::endl;
    std::cout << "BATCH_SIZE: " << dim.batch() << std::endl;
  }
  for (auto &dim : output_dim) {
    std::cout << "OUTPUT_SHAPE : " << dim.channel() << ", " << dim.height()
              << ", " << dim.width() << std::endl;
  }

  try {
    std::cout << "ML_TRAIN_SUMMARY_LAYER" << std::endl;
    model->summarize(std::cout,
                     ml_train_summary_type_e::ML_TRAIN_SUMMARY_LAYER);
    std::cout << "ML_TRAIN_SUMMARY_MODEL" << std::endl;
    model->summarize(std::cout,
                     ml_train_summary_type_e::ML_TRAIN_SUMMARY_MODEL);
    std::cout << "ML_TRAIN_SUMMARY_TENSOR" << std::endl;
    model->summarize(std::cout,
                     ml_train_summary_type_e::ML_TRAIN_SUMMARY_TENSOR);
  } catch (std::exception &e) {
    std::cerr << "uncaught error while training! details: " << e.what()
              << std::endl;
    return 1;
  }

  return status;
}
