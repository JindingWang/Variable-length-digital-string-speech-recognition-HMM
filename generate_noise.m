for i = 1:5
    noise = wgn(16000,1,0)/100;
    audiowrite(['noise_data/', num2str(i), '.wav'], noise, 16000);
end
%str = '15521120199';
%noise = [];
%for i = 1 : length(str)
%    path = ['train_data/', str(i), '/jinhongchen_', str(i), '_1.wav'];
%    fprintf('%s\n', path);
%    audio = audioread(path);
%    noise = [noise; audio];
%end
%audiowrite([str, '_1.wav'], noise, 16000);