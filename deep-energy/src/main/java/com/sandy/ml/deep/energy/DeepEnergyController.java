package com.sandy.ml.deep.energy;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class DeepEnergyController {

    @Autowired
    private DeepEnergyService service;

    @GetMapping("/train-and-predict")
    public String trainAndPredict() {
        try {
            service.trainModel();
            String prediction = service.predict();
            return "Training completed. Prediction for first sequence: " + prediction;
        } catch (Exception e) {
            return "Error: " + e.getMessage();
        }
    }
}