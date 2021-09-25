train_data = dir('train_data');
n=length(train_data);
Miu = [];
Sigma = [];
transition_probability = [];
state = 5;  % 每个HMM所含有的节点数
noise_state = 1;  % noise的HMM所含有的节点数

for i = 1:n
    if strcmp(train_data(i).name,'.') || strcmp(train_data(i).name,'..')
    else
        digital_train_data = dir(['train_data\', train_data(i).name]);
        file_num = length(digital_train_data);
        mfcc_cell = cell(1, file_num-2);
        count = 0;
        for j = 1:file_num
            if strcmp(digital_train_data(j).name,'.') || strcmp(digital_train_data(j).name,'..')
            else
                count = count + 1;
                mfcc_cell{count} = getMFCC(['train_data\', train_data(i).name, '\', digital_train_data(j).name]);
            end
        end
        [digital_Miu, digital_Sigma, log_transition_probability] = HMM(mfcc_cell, state);
        Miu = [Miu; digital_Miu];
        Sigma = [Sigma; digital_Sigma];
        transition_probability = [transition_probability; log_transition_probability];
    end
end

noise_data = dir('noise_data');
n=length(noise_data);
count = 0;
noise_mfcc_cell = cell(1,n-2);
for i = 1:n
    if strcmp(noise_data(i).name,'.') || strcmp(noise_data(i).name,'..')
    else
        count = count + 1;
        noise_mfcc_cell{count} = getMFCC(['noise_data\', noise_data(i).name]);
    end
end
[noise_Miu, noise_Sigma, noise_log_transition_probability] = HMM(noise_mfcc_cell, noise_state);

test_data = dir('test_data');
n=length(test_data);
for i = 1:n
    if strcmp(test_data(i).name,'.') || strcmp(test_data(i).name,'..')
    else
        mfcc = getMFCC(['test_data\',test_data(i).name]);
        word = viterbi_search_backpointer(mfcc, Miu, Sigma, transition_probability, state,...
            noise_Miu, noise_Sigma, noise_log_transition_probability, noise_state);
        fprintf('%s\n',word);
    end
end