filename = 'MissingFeaturesGTZAN.csv';
genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"];

X = zeros(1000, 77);
y = zeros(1000);

index = 1;
for g = 1:numel(genres)
    genre = genres(g);
    folder_path = sprintf('Datasets/GTZAN/audio/%s/*.au', genre);
    files = dir(folder_path);
    for file = files'
        path = sprintf('Datasets/GTZAN/audio/%s/%s', genre, file.name);
        audio = miraudio(path);    
        spectrum = mirspectrum(audio, 'Frame', 23, 0.5);
        
        %Intensity
        rms = mirgetdata(features.dynamics.rms);
        rms_mean = mean(rms);
        rms_std = std(rms);
        rms_dmean = mean(diff(rms));
        rms_dstd = std(diff(rms));
          
        %Pitch              
        ihrate = mirgetdata(mirmean(mirinharmonicity(spectrum)));
                        
        %Timbre
        barkspectrum = mirspectrum(audio, 'Log', 'Bark', 'Frame', 23, 0.5);
        envelope = squeeze(mirgetdata(mirenvelope(barkspectrum)));
        envelopemean = mean(envelope);
        envelopestd = std(envelope);
        
%         flatness = mirgetdata(mirflatness(spectrum));
        
        spreaddata = mirgetdata(mirspread(spectrum));
        spread_nan = spreaddata(~isnan(spreaddata))';
        spreadmean = mean(spread_nan);
        spreadstd = std(spread_nan);
        spreaddiffmean = mean(diff(spread_nan));
        spreaddiffstd = std(diff(spread_nan));
        
        centroid = mirgetdata(mircentroid(spectrum));
        centroiddata = centroid(~isnan(centroid))';
        centroiddiff = diff(centroiddata);
        
        centroidmean = mean(centroiddata);
        centroidstd = std(centroiddata);
        
        centroiddiffmean = mean(centroiddiff);
        centroiddiffstd = std(centroiddiff);
                                   
        %Tonality
        ks = mirkeystrength(spectrum);
        ksmean = mean(mirgetdata(mirmean(ks)));
        ksstd = mirgetdata(mirstd(ks));
        
        tc = mirtonalcentroid(spectrum);
        tcmean = mirgetdata(mirmean(tc));
        tcstd = mirgetdata(mirstd(tc));
        
        hc = mirhcdf(spectrum);
        hcmean = mirgetdata(mirmean(hc));
        hcstd = mirgetdata(mirstd(hc));
        
        %Rhythm
        bs = mirbeatspectrum(audio);
        bsmean = mirgetdata(mirmean(bs));
        bsstd = mirgetdata(mirstd(bs));
        
        bstrength = mirgetdata(mirpeaks(mirevents(audio)));
        bstrengthmean = mean(bstrength);
        bstrengthstd = std(bstrength);
        
        at = mirattacktime(audio);
        atmean = mirgetdata(mirmean(at));
        atstd = mirgetdata(mirstd(at));
        
        tempoAC = mirgetdata(mirtempo(audio, 'Autocor'));
        tempoSpec = mirgetdata(mirtempo(audio, 'Spectrum'));
        
        combined_features = [ihrate, envelopemean, envelopestd, spreadmean, spreadstd, spreaddiffmean, ...
                            spreaddiffstd centroidmean, centroidstd, centroiddiffmean, centroiddiffstd, rot90(tcmean), rot90(tcstd), hcmean, ...
                            hcstd, bsmean, bsstd, bstrengthmean, bstrengthstd, atmean, atstd, tempoAC, tempoSpec];
                        
        NrNaN = sum(isnan(combined_features()));
        if NrNaN > 0            
        end
        
        X(index, :) = combined_features;
        y(index) = genre;
        
        index = index + 1;
    end
end

csvwrite('SAHSRemainingFeatures.csv', X)

    