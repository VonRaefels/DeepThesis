mic1 = 1;
mic2 = 2;
FS = 16000;
S = FS*2;
N = 1000;
load 'Impulse_response_Acoustic_Lab_Bar-Ilan_University_(Reverberation_0.160s)_3-3-3-8-3-3-3/Impulse_response_Acoustic_Lab_Bar-Ilan_University_(Reverberation_0.160s)_3-3-3-8-3-3-3_1m_000.mat';
h0 = impulse_response; 
load 'Impulse_response_Acoustic_Lab_Bar-Ilan_University_(Reverberation_0.160s)_3-3-3-8-3-3-3/Impulse_response_Acoustic_Lab_Bar-Ilan_University_(Reverberation_0.160s)_3-3-3-8-3-3-3_1m_030.mat';
h30 = impulse_response;
factor = 0.35;
d = rdir('data_timit/TRAIN/**/*.WAV');
speech = zeros(numel(d), S);
for k=1:numel(d)
    [y Fs] = audioread(d(k).name);
    l = size(y, 1);
    if l > S
        l = S;
    end
    speech(k,1:l) = y(1:l)';
end
disp('writing data')
indeces = zeros(2, N);
for i = 1:N
    j = randi([1 numel(d)]);
    l = randi([1 numel(d)]);
    dir_i = d(j).folder;
    dir_j = d(l).folder;
    while strcmp(dir_i, dir_j) == true || ~isempty(mfind(indeces, [j ; l]))
        l = randi([1 numel(d)]);
        dir_j = d(l).folder;
        disp('same dir');
        disp(dir_i);
        disp(dir_j);
    end
    indeces(:,i) = [j l]';
    target = speech(j,:);
    echo = factor*speech(l,:);
    %target = wgn(FS, 1, 0);
    %echo = factor*wgn(FS, 1, 0);
    %soundsc(n, FS);

    y11 = conv(target, h0(:,mic1));
    y12 = conv(target, h0(:,mic2));

    y21 = conv(echo, h30(:,mic1));
    y22 = conv(echo, h30(:,mic2));

    out1 = y11 + y21;
    out2 = y12 + y22;

%     out1 = out1 / max(abs(out1));
%     out2 = out2 / max(abs(out2));
%     target = target / max(abs(target));
%     echo = echo / max(abs(echo));
%     y11 = y11 / max(abs(y11));
%     y22 = y22 / max(abs(y22));
%     y12 = y12 / max(abs(y12));
%     y21 = y21 / max(abs(y21));

    mkdir(strcat('/home/jorge/Documents/data/', num2str(i)));
    audiowrite(sprintf('/home/jorge/Documents/data/%d/mix1.wav', i), out1(1:S), FS);
    audiowrite(sprintf('/home/jorge/Documents/data/%d/mix2.wav',i), out2(1:S), FS);
    %audiowrite(sprintf('/home/jorge/Documents/data/%d/target.wav', i), target, FS);
    audiowrite(sprintf('/home/jorge/Documents/data/%d/echo.wav', i), echo, FS);
    %audiowrite(sprintf('/home/jorge/Documents/data/%d/mic1_target.wav', i), y11(1:S), FS);
    %audiowrite(sprintf('/home/jorge/Documents/data/%d/mic2_target.wav', i), y12(1:S), FS);
    %audiowrite(sprintf('/home/jorge/Documents/data/%d/mic1_echo.wav', i), y21(1:S), FS);
    %audiowrite(sprintf('/home/jorge/Documents/data/%d/mic2_echo.wav', i), y22(1:S), FS);
    
    
    % Remember to change number of channels
    mix1_feats = my_features_AmsRastaplpMfccGf(out1(1:S));
    csvwrite(sprintf('/home/jorge/Documents/data/%d/mix1.out', i), mix1_feats);
    mix2_feats = my_features_AmsRastaplpMfccGf(out2(1:S));
    csvwrite(sprintf('/home/jorge/Documents/data/%d/mix2.out', i), mix2_feats);
    echo_feats = my_features_AmsRastaplpMfccGf(echo(1:S));
    csvwrite(sprintf('/home/jorge/Documents/data/%d/echo.out', i), echo_feats);
    
    is_wiener_mask = 0;
    db = 0;
    mix1_ibm = just_ibm( out1(1:S), y11(1:S), y21(1:S), 128, is_wiener_mask, db);
    csvwrite(sprintf('/home/jorge/Documents/data/%d/ibm.out', i), mix1_ibm);
    
end
disp('finished!');
