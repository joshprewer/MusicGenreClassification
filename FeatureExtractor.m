filename = 'RemainingFeaturesISMIR.csv';
genres = ["classical",'electronic', 'jazz', 'metal', 'pop', 'punk', 'rock', 'world'];

X = zeros(1427, 19);
y = zeros(1427);

index = 1;
for g = 1:numel(genres)
    genre = genres(g);
    folder_path = sprintf('Datasets/ismir04_genre/audio/training/%s/*.mp3', genre);  
    files = dir(folder_path);    
    for file = files'
        path = sprintf('Datasets/ismir04_genre/audio/training/%s/%s', genre, file.name);
        info = audioinfo(path);
        audio = miraudio(path, 'Extract', (info.Duration / 2) - 15, (info.Duration / 2) + 15, 'Sampling', 22050);    
        spectrum = mirspectrum(audio, 'Frame', 23, 0.5);
        
        %Tonality
        ihrate = mirgetdata(mirmean(mirinharmonicity(spectrum)));
                                              
        hc = mirgetdata(mirhcdf(audio, 'Frame'));
        hcmean = mean(hc);
        hcstd = std(hc);
        hcdiffmean = mean(diff(hc));
        hcdiffstd = std(diff(hc));        
        
        %Rhythm              
        bstrength = mirgetdata(mirpeaks(mirevents(audio)));
        bstrengthmean = mean(bstrength);
        bstrengthstd = std(bstrength);
        bstrengthdiffmean = mean(diff(bstrength));
        bstrengthdiffstd = std(diff(bstrength));
        
        at = mirgetdata(mirattacktime(audio));
        atmean = mean(at);
        atstd = std(at);
        atdiffmean = mean(diff(at));
        atdiffstd = std(diff(at));
        
        pc = mirpulseclarity(audio, 'Frame');
        pcmean = mirgetdata(mirmean(pc));
        pcstd = mirgetdata(mirstd(pc));
        pcdiffmean = mean(diff(mirgetdata(pc)));
        pcdiffstd = std(diff(mirgetdata(pc)));
        
        tempoAC = mirgetdata(mirtempo(audio, 'Autocor'));
        tempoSpec = mirgetdata(mirtempo(audio, 'Spectrum'));
        
        combined_features = [ihrate, hcmean, hcstd, hcdiffmean, hcdiffstd ...
                             bstrengthmean, bstrengthstd, bstrengthdiffmean, bstrengthdiffstd ...
                             atmean, atstd, atdiffmean, atdiffstd ...
                             pcmean, pcstd, pcdiffmean, pcdiffstd ...
                             tempoAC, tempoSpec];
        
        NrNaN = sum(isnan(combined_features()));
        if NrNaN > 0
        end
        
        X(index, :) = combined_features;
        y(index) = genre;
        
        index = index + 1;
    end
end

csvwrite(filename, X)