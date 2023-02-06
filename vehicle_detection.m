% Nama anggota 1 : Zidan Hidayat Al-Kautsar / 13219023
% Nama anggota 2 : Diaz Zaid Abdurrahman / 13219028
% Nama anggota 3 : M Sandi Putra Pratama / 13219014

%% Menggunanakan Faster R-CNN untuk mendeteksi objek berupa mobil

%% Download Package Image  processing 
%% Deep learning Toolbox
%% Computer Vision Toolbox

% Load vehicle data set
data = load('fasterRCNNVehicleTrainingData.mat');
vehicleDataset = data.vehicleTrainingData;

% Pisahkan training 60 % dan test set sisanya
idx = floor(0.6 * height(vehicleDataset));
trainingData = vehicleDataset(1:idx,:);
testData = vehicleDataset(idx:end,:);

%% Membuat Convolutional Neural Network (CNN)

% Membuat image input layer
inputLayer = imageInputLayer([28 28 3]);

%% Membuat network layer tengah
% Membuat layer tengah sebagai region of Pooling
middleLayers = [            
    convolution2dLayer([3 3], 32,'Padding',1) 
    reluLayer()     
    convolution2dLayer([3 3], 32,'Padding',1)  
    reluLayer() 
    maxPooling2dLayer(3, 'Stride',2)    
    ];

%% Layer akhir untuk CNN terdiri dari fully connecter layer serta softmax layer

finalLayers = [
    % FC layer dengan 64 output neuron
    fullyConnectedLayer(64)
    reluLayer

    % Menambah fc layer terakhir yang output nya harus sesuai dengan width
    % dataset
    fullyConnectedLayer(width(vehicleDataset))

    % Menambah siftmax dan classification
    softmaxLayer
    classificationLayer
    ];

%%
% Menggabungkan semua input layer
layers = [
    inputLayer
    middleLayers
    finalLayers
];


%% Mengonfigurasi training options
% step 1 dan 2 untuk blok regional proposal network dan Fast R-CNN
% step 3  dan 4 untuk menggabungkan network  sehingga bisa dilakukan
% deteksi teradap satu network/jaringan tersebut

% Options untuk step 1.
optionsStage1 = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 1, ...
    'InitialLearnRate', 1e-3, ...
    'CheckpointPath', tempdir);

% Options untuk step 2.
optionsStage2 = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 1, ...
    'InitialLearnRate', 1e-3, ...
    'CheckpointPath', tempdir);

% Options untuk step 3.
optionsStage3 = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 1, ...
    'InitialLearnRate', 1e-3, ...
    'CheckpointPath', tempdir);

% Options untuk step 3.
optionsStage4 = trainingOptions('sgdm', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 1, ...
    'InitialLearnRate', 1e-3, ...
    'CheckpointPath', tempdir);

options = [
    optionsStage1
    optionsStage2
    optionsStage3
    optionsStage4
    ];

%% Training Faster R-CNN
doTrainingAndEval = false;

if doTrainingAndEval
    rng(0);
    % Melatik Faster R-CNN detector 
    % Range negative 0 - 0.3
    % Range Positive 0.5 - 1
    detector = trainFasterRCNNObjectDetector(trainingData, layers, options, ...
        'NegativeOverlapRange', [0 0.3], ...
        'PositiveOverlapRange', [0.5 1], ...
        'NumRegionsToSample', [256 128 256 128], ...
        'BoxPyramidScale', 1.2);
else
    %load detector sebagai detector awal dari dataset sebelumnya
    detector = data.detector;
end

%% Pengetesan gambar
%ubah file pada im read jika ingin menggunakan gambar lain
I = imread("highwaycar.jpg");
[bboxes, scores] = detect(detector, I);
I = insertObjectAnnotation(I, 'rectangle', bboxes, scores);
figure
imshow(I)

%%
if doTrainingAndEval
    % Melakukan detector pada salah satu gambar 
    resultsStruct = struct([]);
    for i = 1:height(testData)
        
        I = imread(testData.imageFilename{i});
        
        % Melakukan detector
        [bboxes, scores, labels] = detect(detector, I);
        resultsStruct(i).Boxes = bboxes;
        resultsStruct(i).Scores = scores;
        resultsStruct(i).Labels = labels;
    end
    
    % Konversi ke hasil ke tabel
    results = struct2table(resultsStruct);
else
    results = data.results;
end

hasilEkspetasi = testData(:, 2:end);

%  Evaluasi objek detektor dengan average precision metric
[ap, recall, precision] = evaluateDetectionPrecision(results, hasilEkspetasi);

% Plot kurva
figure
plot(recall, precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f', ap))





