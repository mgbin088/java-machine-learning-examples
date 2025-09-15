package com.sandy.ml.deep.energy;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.timeseries.Forecast;
import ai.djl.timeseries.TimeSeriesData;
import ai.djl.timeseries.dataset.FieldName;
import ai.djl.timeseries.model.deepar.DeepARNetwork;
import ai.djl.timeseries.translator.DeepARTranslator;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.dataset.Record;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Adam;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.Batchifier;
import ai.djl.translate.TranslateException;
import ai.djl.util.PairList;
import ai.djl.util.Progress;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.util.*;

@Service
public class DeepEnergyService {

    private static final int NUM_FACTORIES = 3;
    private static final int NUM_MEDIA = 3;
    private static final int NUM_PRODUCT = 2;
    private static final int HISTORY_LENGTH = 24;
    private static final int PREDICTION_LENGTH = 12;
    private static final String FREQ = "M";
    private static final LocalDateTime START_TIME = LocalDateTime.of(2023, 1, 1, 0, 0);

    private ZooModel<TimeSeriesData, Forecast> model;
    private static String currentPath;

    public void trainModel() throws IOException, TranslateException {
        try (NDManager manager = NDManager.newBaseManager("MXNet")) {
            // 模拟数据
            List<TimeSeriesData> trainDataList = generateSimulatedData(manager, HISTORY_LENGTH);

            // 自定义 Dataset
            MyTimeSeriesDataset trainDataset = new MyTimeSeriesDataset(trainDataList, HISTORY_LENGTH, true);

            // 配置 DeepAR Estimator
            List<Integer> cardinality = Arrays.asList(NUM_FACTORIES, NUM_MEDIA);
            DeepARNetwork trainingNetwork = new DeepARNetwork.Builder()
                    .setPredictionLength(PREDICTION_LENGTH)
                    .optContextLength(HISTORY_LENGTH)
                    .setFreq(FREQ)
                    .optNumLayers(2)
                    .buildTrainingNetwork();
            // 构造 Model
            Model model = Model.newInstance("deepar");
            model.setBlock(trainingNetwork);
            // 构造训练配置
            DefaultTrainingConfig trainConfig = new DefaultTrainingConfig(Loss.l2Loss())
                    .optOptimizer(Adam.builder().build())
                    .addTrainingListeners(TrainingListener.Defaults.logging())
                    .optDevices(manager.getEngine().getDevices());
            // 创建 Trainer
            Trainer trainer = model.newTrainer(trainConfig);
            // 初始化输入形状
            trainer.initialize(new Shape(1, HISTORY_LENGTH, NUM_PRODUCT), new Shape(1, HISTORY_LENGTH, NUM_PRODUCT), new Shape(1, cardinality.size()));
            // 训练模型
            EasyTrain.fit(trainer, 5, trainDataset, null);
            // 保存模型
            String modelId = UUID.randomUUID().toString();
            String path = "models/deepar_" + modelId;
            model.save(Paths.get(path), "deepar");
            currentPath=path;
            // 清理资源
            trainer.close();
            model.close();
        }
    }

    public String predict() throws TranslateException {
        try (NDManager manager = NDManager.newBaseManager("MXNet")) {
            Map<String, Object> arguments= new HashMap<>();
            arguments.put("prediction_length", PREDICTION_LENGTH);
            arguments.put("freq", FREQ);
            arguments.put("context_length", HISTORY_LENGTH);
            arguments.put("num_layers", 2);
            DeepARTranslator translator = DeepARTranslator.builder(arguments).build();
            Criteria<TimeSeriesData, Forecast> criteria = Criteria.builder()
                    .setTypes(TimeSeriesData.class, Forecast.class)
                    .optTranslator(translator)
                    .optProgress(new ProgressBar())
                    .build();
            ZooModel<TimeSeriesData, Forecast> currentModel = criteria.loadModel();
            currentModel.load(Paths.get(currentPath), "deepar");
            Predictor<TimeSeriesData, Forecast> predictor = currentModel.newPredictor();
            // 模拟未来动态特征（历史 + 未来12个月）
            NDArray futureDynamic = manager.randomNormal(new Shape(NUM_PRODUCT, HISTORY_LENGTH + PREDICTION_LENGTH));
            // 示例：第一个序列的输入
            TimeSeriesData input = new TimeSeriesData(10);
            input.setStartTime(START_TIME);
            input.setField(FieldName.FEAT_DYNAMIC_REAL, futureDynamic.transpose()); // 动态特征 (时间 x 特征)
            input.setField(FieldName.FEAT_STATIC_CAT, manager.create(new long[]{0, 0})); // 静态: 工厂0, 介质0
            Forecast forecast = predictor.predict(input);
            return Arrays.toString(forecast.mean().toFloatArray()); // 返回均值预测
        } catch (Throwable e) {
            e.printStackTrace();
            return "Error during prediction: " + e.getMessage();
        }
    }

    private List<TimeSeriesData> generateSimulatedData(NDManager manager, int length) {
        List<TimeSeriesData> dataList = new ArrayList<>();
        Random random = new Random(42);
        for (int f = 0; f < NUM_FACTORIES; f++) {
            for (int m = 0; m < NUM_MEDIA; m++) {
                TimeSeriesData data = new TimeSeriesData(10);
                data.setStartTime(START_TIME);
                float[] targetArr = new float[length];
                for (int i = 0; i < length; i++) {
                    targetArr[i] = 100 + random.nextFloat() * 10;
                }
                data.setField(FieldName.TARGET, manager.create(targetArr));
                // 动态特征: NUM_PRODUCT x length
                NDArray dynamic = manager.randomNormal(new Shape(NUM_PRODUCT, length));
                data.setField(FieldName.FEAT_DYNAMIC_REAL, dynamic.transpose()); // 转置为 length x NUM_PRODUCT
                // 静态特征: [工厂ID, 介质ID]
                data.setField(FieldName.FEAT_STATIC_CAT, manager.create(new long[]{f, m}));
                dataList.add(data);
            }
        }
        return dataList;
    }

    private TrainingConfig getTrainingConfig() {
        return new DefaultTrainingConfig(Loss.l1Loss())
                .optOptimizer(Adam.builder().build());
    }

    // 自定义 Dataset (扩展 RandomAccessDataset 处理多序列)
    private static class MyTimeSeriesDataset extends RandomAccessDataset {
        private final List<TimeSeriesData> dataList;
        private final int contextLength;
        private final boolean isTrain;

        public MyTimeSeriesDataset(RandomAccessDataset.BaseBuilder<?> builder,List<TimeSeriesData> dataList, int contextLength, boolean isTrain) {
            super(builder);
            this.dataList = dataList;
            this.contextLength = contextLength;
            this.isTrain = isTrain;
        }

        @Override
        protected long availableSize() {
            return dataList.size();
        }

        @Override
        public Record get(NDManager ndManager, long l) throws IOException {
            return null;
        }

        @Override
        public Iterable<Batch> getData(NDManager manager) throws IOException {
            return () -> new Iterator<Batch>() {
                private final Iterator<TimeSeriesData> iterator = dataList.iterator();
                @Override
                public boolean hasNext() {
                    return iterator.hasNext();
                }

                @Override
                public Batch next() {
                    List<NDList> data = new ArrayList<>();
                    List<NDList> labels = new ArrayList<>();
                    for (int i = 0; i < MyTimeSeriesDataset.super.sampler.getBatchSize(); i++) {
                        if (iterator.hasNext()) {
                            TimeSeriesData tsData = iterator.next();
                            PairList<NDArray, Shape> batch =
                                    DeepARNetwork.b(manager, Collections.singletonList(tsData), isTrain, contextLength, PREDICTION_LENGTH);
                            data.add(batch.getKey());
                            labels.add(batch.getValue());
//                                    Feature.batchify(manager, Collections.singletonList(tsData), isTrain, contextLength, PREDICTION_LENGTH);
//                            data.add(batch.getKey());
//                            labels.add(batch.getValue());
                        }
                    }
                    return new Batch(manager, data.toArray(new NDList[0]), labels.toArray(new NDList[0]), MyTimeSeriesDataset.super.sampler.getBatchSize(), Batchifier.STACK, Batchifier.STACK, 0, 0);
                }
            };
        }

        @Override
        public void prepare(Progress progress) throws IOException, TranslateException {

        }
    }
}