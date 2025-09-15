package com.sandy.ml.energy.predict.service;

import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.timeseries.Forecast;
import ai.djl.timeseries.TimeSeriesData;
import ai.djl.timeseries.dataset.FieldName;
import ai.djl.timeseries.distribution.output.DistributionOutput;
import ai.djl.timeseries.distribution.output.NegativeBinomialOutput;
import ai.djl.timeseries.model.deepar.DeepARNetwork;
import ai.djl.timeseries.translator.DeepARTranslator;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Adam;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;
import com.sandy.ml.energy.predict.model.DeepArConfig;
import com.sandy.ml.energy.predict.model.DeepArData;
import com.sandy.ml.energy.predict.model.DeepArModel;
import com.sandy.ml.energy.predict.model.MultipleTimeSeriesDataset;
import com.sandy.ml.energy.predict.repository.DeepArModelRepository;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

@Service
@Slf4j
public class DeepArDataService {
    // 训练进度存储，key为sessionId，value为进度描述
    private final ConcurrentHashMap<String, String> trainProgressMap = new ConcurrentHashMap<>();

    @Autowired
    private DeepArModelRepository modelRepository;
    @Autowired
    private DeepArConfigService deepArConfigService;


    // 合并上传和训练
    public void uploadAndTrain(MultipartFile file, DeepArConfig config, String sessionId) {
        trainProgressMap.put(sessionId, "上传文件中...");
        List<DeepArData> dataList = parseCsv(file, config);
        trainProgressMap.put(sessionId, "上传文件完成，开始数据预处理...");
        // ��据预处理（示例：归一化、缺失值处理）
        preprocessData(dataList, config);
        trainProgressMap.put(sessionId, "训练中：开始训练...");
        // 模型训练
        train(dataList, config, sessionId);
        trainProgressMap.put(sessionId, "训练完成");
    }

    // 数据预处理
    private void preprocessData(List<DeepArData> dataList, DeepArConfig config) {
        for (DeepArData data : dataList) {
            List<Double> target = data.getTarget();
            if (target != null && !target.isEmpty()) {
                double max = target.stream().mapToDouble(Double::doubleValue).max().orElse(1.0);
                double min = target.stream().mapToDouble(Double::doubleValue).min().orElse(0.0);
                if (max != min) {
                    target.replaceAll(v -> (v - min) / (max - min));
                }
            }
        }
    }

    // 解析CSV文件为TimeSeriesDataParameter对象列表
    public List<DeepArData> parseCsv(MultipartFile file, DeepArConfig config) {
        List<DeepArData> dataList = new ArrayList<>();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(file.getInputStream()))) {
            String line;
            boolean isFirst = true;
            String[] headers = null;
            while ((line = reader.readLine()) != null) {
                if (isFirst) {
                    headers = line.split(",");
                    isFirst = false;
                    continue;
                }
                String[] values = line.split(",");
                DeepArData data = new DeepArData();
                // 按照 config 字段映射解析
                for (int i = 0; i < headers.length; i++) {
                    String field = headers[i].trim();
                    String value = values[i].trim();
                    // 解析主字段
                    if (field.equalsIgnoreCase("start")) {
                        data.setStart(value);
                    } else if (field.equalsIgnoreCase("itemId")) {
                        data.setItemId(value);
                    } else if (field.equalsIgnoreCase("target")) {
                        // 支持逗号分隔的数序列
                        List<Double> target = new ArrayList<>();
                        for (String v : value.split(";")) {
                            try { target.add(Double.parseDouble(v)); } catch (Exception ignore) {}
                        }
                        data.setTarget(target);
                    }
                    // 静态类别特征
                    if (config.getStaticCatFields() != null && Arrays.asList(config.getStaticCatFields()).contains(field)) {
                        if (data.getFeatStaticCat() == null) data.setFeatStaticCat(new ArrayList<>());
                        data.getFeatStaticCat().add(value);
                    }
                    // 静态类别特征
                    if (config.getStaticRealFields() != null && Arrays.asList(config.getStaticRealFields()).contains(field)) {
                        if (data.getFeatStaticReal() == null) data.setFeatStaticReal(new ArrayList<>());
                        try { data.getFeatStaticReal().add(Double.parseDouble(value)); } catch (Exception ignore) {}
                    }
                    // 动态数值特征
                    if (config.getDynamicRealFields() != null && Arrays.asList(config.getDynamicRealFields()).contains(field)) {
                        if (data.getFeatDynamicReal() == null) data.setFeatDynamicReal(new ArrayList<>());
                        List<Double> dynReal = new ArrayList<>();
                        for (String v : value.split(";")) {
                            try { dynReal.add(Double.parseDouble(v)); } catch (Exception ignore) {}
                        }
                        data.getFeatDynamicReal().add(dynReal);
                    }
                    // 动态类别特征
                    if (config.getDynamicCatFields() != null && Arrays.asList(config.getDynamicCatFields()).contains(field)) {
                        if (data.getFeatDynamicCat() == null) data.setFeatDynamicCat(new ArrayList<>());
                        String[] vs = value.split(";");
                        List<Integer> dynCat = new ArrayList<>();
                        for (String v : vs) {
                            dynCat.add(Integer.parseInt(v));
                        }
                        data.getFeatDynamicCat().add(dynCat);
                    }
                }
                dataList.add(data);
            }
        } catch (Exception e) {
            log.error("ParseCsv Error",e);
        }
        return dataList;
    }

    // 训练逻辑
    public void train(List<DeepArData> dataList, DeepArConfig config, String sessionId) {
        if (sessionId != null) trainProgressMap.put(sessionId, "训练中：模型拟合...");
        String modelId = UUID.randomUUID().toString();
        String path = "models/deepar_" + modelId;
        try (NDManager manager = NDManager.newBaseManager("MXNet")) {
            // 1. 构造 DeepARNetwork
            int staticCatNum = config.getStaticCatFields() != null ? config.getStaticCatFields().length : 0;
            int staticRealNum = config.getStaticRealFields() != null ? config.getStaticRealFields().length : 0;
            int dynamicRealNum = config.getDynamicRealFields() != null ? config.getDynamicRealFields().length : 0;
            int dynamicCatNum = config.getDynamicCatFields() != null ? config.getDynamicCatFields().length : 0;
            DistributionOutput distributionOutput = new NegativeBinomialOutput();
            DeepARNetwork trainingNetwork = new DeepARNetwork.Builder()
                    .setPredictionLength(config.getPredictionLength())
                    .setFreq(config.getFreq())
                    .optContextLength(config.getContextLength())
                    .optNumLayers(config.getNumLayers())
                    .optHiddenSize(config.getHiddenSize())
                    .optDropRate(config.getDropoutRate())
                    .optEmbeddingDimension(List.of(config.getEmbeddingDimension()))
                    .optUseFeatStaticCat(config.isUseFeatStaticCat())
                    .optUseFeatStaticReal(config.isUseFeatStaticReal())
                    .optUseFeatDynamicReal(config.isUseFeatDynamicReal())
                    .optDistrOutput(distributionOutput)
                    .buildTrainingNetwork();
            // 2. 构造 Model
            Model model = Model.newInstance("deepar");
            model.setBlock(trainingNetwork);
            // 3. 构造训练数据集
            Dataset trainSet = getDataset(dataList, config, manager,trainingNetwork);
            // 4. 构造训练配置
            DefaultTrainingConfig trainConfig = new DefaultTrainingConfig(Loss.l2Loss())
                    .optOptimizer(Adam.builder().build())//.setLearningRate(config.getLearningRate())
                    .addTrainingListeners(TrainingListener.Defaults.logging())
                    .optDevices(manager.getEngine().getDevices());
            // 5. 创建 Trainer
            Trainer trainer = model.newTrainer(trainConfig);
            // 6. 初始化输入形状
            int batchSize = config.getBatchSize();
            int contextLength = config.getContextLength();
            int featureNum = 1 + staticCatNum + staticRealNum + dynamicRealNum + dynamicCatNum; // 1为target
            Shape shape = new Shape(batchSize, contextLength, featureNum);
            trainer.initialize(shape);
            // 7. 训练模型
            int epochs = config.getEpochs();
            EasyTrain.fit(trainer, epochs, trainSet, null);
            // 8. 保存模型
            model.save(Paths.get(path), "deepar");
            // 9. 清理资源
            trainer.close();
            model.close();
        } catch (Exception e) {
            e.printStackTrace();
            if (sessionId != null) trainProgressMap.put(sessionId, "训练失败:" + e.getMessage());
            return;
        }
        if (sessionId != null) trainProgressMap.put(sessionId, "训练中：保存模型...");
        DeepArModel deepArModel = DeepArModel.builder()
                .code("Model " + modelId)
                .path(path)
                .crps(0.0f)
                .rmse(0.0f)
                .createdAt(LocalDateTime.now())
                .build();
        modelRepository.save(deepArModel);
        if (sessionId != null) trainProgressMap.put(sessionId, "训练完成");
    }

    // 构造训练数据集
    private Dataset getDataset(List<DeepArData> dataList, DeepArConfig config, NDManager manager, DeepARNetwork trainingNetwork) {
        List<NDArray> targetList = new ArrayList<>();
        List<NDArray> staticCatList = new ArrayList<>();
        List<NDArray> staticRealList = new ArrayList<>();
        List<NDArray> dynamicRealList = new ArrayList<>();
        List<NDArray> dynamicCatList = new ArrayList<>();
        for (DeepArData data : dataList) {
            targetList.add(manager.create(data.getTarget().stream().mapToDouble(Double::doubleValue).toArray()));
            if (config.isUseFeatStaticCat() && data.getFeatStaticCat() != null) {
                staticCatList.add(manager.create(data.getFeatStaticCat().stream().mapToInt(String::hashCode).toArray()));
            }
            if (config.isUseFeatStaticReal() && data.getFeatStaticReal() != null) {
                staticRealList.add(manager.create(data.getFeatStaticReal().stream().mapToDouble(Double::doubleValue).toArray()));
            }
            if (config.isUseFeatDynamicReal() && data.getFeatDynamicReal() != null) {
                double[][] dynReals= new double[data.getFeatDynamicReal().size()][data.getFeatDynamicReal().get(0).size()];
                for (int i = 0; i < data.getFeatDynamicReal().size(); i++) {
                    List<Double> dyn = data.getFeatDynamicReal().get(i);
                    for (int j = 0; j < dyn.size(); j++) {
                        dynReals[i][j]=dyn.get(j);
                    }
                }
                dynamicRealList.add(manager.create(dynReals));
            }
            if (config.isUseFeatDynamicCat() && data.getFeatDynamicCat() != null) {
                int[][] dynCats= new int[data.getFeatDynamicCat().size()][data.getFeatDynamicCat().get(0).size()];
                for (int i = 0; i < data.getFeatDynamicCat().size(); i++) {
                    List<Integer> dyn = data.getFeatDynamicCat().get(i);
                    for (int j = 0; j < dyn.size(); j++) {
                        dynCats[i][j]=dyn.get(j);
                    }
                }
                dynamicCatList.add(manager.create(dynCats));
            }
        }

        // 这里只返回 targetList，实际可扩展为自定义 Dataset
        return new MultipleTimeSeriesDataset(
                null,
                targetList,
                staticRealList,
                dynamicRealList,
                dynamicCatList
        );
    }


    // 查询所有模型信息
    public List<DeepArModel> getModelList() {
        return modelRepository.findAll();
    }

    // 启用指��模型
    public void enableModel(String modelId) {
        DeepArModel model = modelRepository.findById(Long.parseLong(modelId)).orElse(null);
        if (model != null) {
            model.setEnabled(true);
            model.setUpdatedAt(LocalDateTime.now());
            modelRepository.save(model);
        }
    }

    // 禁用指定���型
    public void disableModel(String modelId) {
        DeepArModel model = modelRepository.findById(Long.parseLong(modelId)).orElse(null);
        if (model != null) {
            model.setEnabled(false);
            model.setUpdatedAt(LocalDateTime.now());
            modelRepository.save(model);
        }
    }

    // 删除指定模型
    public void deleteModel(String modelId) {
        modelRepository.deleteById(Long.parseLong(modelId));
    }

    // 预测逻辑
    public Object predict(DeepArData param, DeepArConfig config) {
        try (NDManager manager = NDManager.newBaseManager()) {
            Map<String, Object> arguments = new ConcurrentHashMap<>();
            arguments.put("prediction_length", config.getPredictionLength());
            arguments.put("context_length", config.getContextLength());
            DeepARTranslator translator = DeepARTranslator.builder(arguments).build();
            Criteria<TimeSeriesData, Forecast> criteria = Criteria.builder()
                    .setTypes(TimeSeriesData.class, Forecast.class)
                    .optModelUrls("https://resources.djl.ai/test-models/mxnet/timeseries/deepar.zip")
                    .optTranslator(translator)
                    .optProgress(new ProgressBar())
                    .build();
            ZooModel<TimeSeriesData, Forecast> model = deepArConfigService.createDeepARModel(deepArConfigService.findById(1L).orElseThrow()); // 假设使用 ID=1 的配置
            model.load(Paths.get(modelRepository.findByEnabled(true).getPath()));
            try (
                 Predictor<TimeSeriesData, Forecast> predictor = model.newPredictor();
                 NDArray target = manager.create(param.getTarget().stream().mapToDouble(Double::doubleValue).toArray())) {
                TimeSeriesData tsData = new TimeSeriesData(1);
                tsData.setStartTime(LocalDateTime.parse(param.getStart().replace(" ", "T")));
                tsData.setField(FieldName.TARGET, target);
                Forecast forecast = predictor.predict(tsData);
                return forecast.mean().toFloatArray();
            }
        } catch (ModelException | TranslateException | IOException | RuntimeException e) {
            e.printStackTrace();
            return "预测失败:" + e.getMessage();
        }
    }

    // 查��训练进度
    public String getTrainProgress(String sessionId) {
        return trainProgressMap.getOrDefault(sessionId, "无此训练任务");
    }
}
