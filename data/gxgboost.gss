new;

library gxgboost;

_features = "class"$|"cap_shape"$|"cap_surface"$|"cap_color"$|"bruises"$|"odor"$|"gill_attachment"$|"gill_spacing"$|
            "gill_size"$|"gill_color"$|"stalk_shape"$|"stalk_root"$|"stalk_surface_above_ring"$|"stalk_surface_below_ring"$|
             "stalk_color_above_ring"$|"stalk_color_below_ring"$|"veil_type"$|"veil_color"$|"ring_number"$|"ring_type"$|
             "spore_print_color"$|"population"$|"habitat";
             
features = strjoin("cat("$+_features'$+")", " + ");
data = loadd(__FILE_DIR$+"agaricus-lepiota.csv", features);
data[.,1] = (data[.,1] .== 2);

{ train_data, test_data } = nfolds(data, 0.2);

struct XGBTree ctl;
ctl = xgbcreatectl("tree");
ctl.params.eta = 1.0;
ctl.params.gamma = 1.0;
ctl.params.min_child_weight = 1;
ctl.params.max_depth = 3;

ctl.learning.num_round = 2;
ctl.learning.objective = "binary:logistic";

model = xgbtrain(train_data, ctl);
print xgbpredict(model, test_data);

proc (2) = nfolds(data, test_percentage);
    local selection, n, chunk, train_data, test_data;
    chunk = floor(rows(data) * test_percentage);
    n = floor(rndu(1,1)*(1/test_percentage));
    
    selection = seqa(n*chunk+1, 1, chunk);
    
    train_data = data[selection,.];
    test_data = delrows(data, selection);
    
    retp(train_data, test_data);
endp;

