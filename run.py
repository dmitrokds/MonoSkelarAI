import os
import pandas as pd
from xgboost import XGBClassifier
from dataset import create_dataset
import json

# load test files
test_users = pd.read_csv("test_users.csv")
test_tx = pd.read_csv("test_transactions.csv")

# build test dataset
test_dataset = create_dataset(test_tx, test_users).copy()

# keep ids
test_ids = test_dataset["id_user"].copy()

# use same feature columns as in train
feature_cols = [c for c in test_dataset.columns if c not in ["id_user", "is_fraud"]]
X_test = test_dataset[feature_cols].copy()

# fill categorical
cat_cols = X_test.select_dtypes(include=["object", "string"]).columns.tolist()
for col in cat_cols:
    X_test[col] = X_test[col].fillna("missing").astype(str)

# one hot
X_test = pd.get_dummies(X_test, columns=cat_cols, dummy_na=False)

# your saved train columns
train_feature_cols = ['transaction_number', 'success_count', 'fail_count', 'antifraud_count', 'amount_mean', 'amount_std', 'amount_min', 'amount_max', 'unique_card_mask_hash_count', 'unique_card_holder_norm_count', 'unique_error_group_count', 'reg_vs_card_country_match_rate', 'reg_vs_payment_country_match_rate', 'card_vs_payment_country_match_rate', 'first_transactions_after_reg_sec', 'last_transactions_after_reg_sec', 'from_last_transaction_mean', 'fail_rate', 'success_rate', 'fraud_error_rate', 'fails_per_hour_active', 'country_contradiction_score', 'fail_before_success_pressure', 'card_pressure_contradiction', 'geo_card_fail_interaction', 'geo_holder_interaction', 'holder_per_card_ratio', 'card_per_holder_ratio', 'holder_norm_change_count', 'transaction_type_change_count', 'payment_country_change_count', 'transaction_type_fail_interaction', 'fs_transactions_count', 'fs_fail_count', 'fs_fraud_error_count', 'fs_fail_rate', 'fs_fraud_error_rate', 'last10_transactions_count', 'last10_fail_count', 'last10_success_count', 'last10_fraud_error_count', 'last10_unique_cards', 'last10_unique_holders', 'last10_unique_transactions_types', 'last10_fail_rate', 'last10_fraud_error_rate', 'last20_transactions_count', 'last20_fail_count', 'last20_success_count', 'last20_fraud_error_count', 'last20_unique_cards', 'last20_unique_holders', 'last20_unique_transactions_types', 'last20_fail_rate', 'last20_fraud_error_rate', 'last10_burst_pressure', 'last20_burst_pressure', 'max_fails_in_1h', 'max_unique_cards_in_1h', 'max_fails_in_6h', 'max_unique_cards_in_6h', 'max_fails_in_24h', 'max_unique_cards_in_24h', 'is_orgainc', 'top_error_group_3ds', 'top_error_group_card_problem', 'top_error_group_cvv', 'top_error_group_dnh', 'top_error_group_fraud', 'top_error_group_insuff', 'top_error_group_invalid', 'top_error_group_issuer_decline', 'top_error_group_limit', 'top_error_group_none', 'top_error_group_other', 'top_error_group_technical', 'top_transaction_type_apple-pay', 'top_transaction_type_card_init', 'top_transaction_type_card_recurring', 'top_transaction_type_google-pay', 'top_transaction_type_resign', 'top_payment_country_afghanistan', 'top_payment_country_albania', 'top_payment_country_algeria', 'top_payment_country_american samoa', 'top_payment_country_andorra', 'top_payment_country_angola', 'top_payment_country_anguilla', 'top_payment_country_antigua and barbuda', 'top_payment_country_argentina', 'top_payment_country_armenia', 'top_payment_country_aruba', 'top_payment_country_australia', 'top_payment_country_austria', 'top_payment_country_azerbaijan', 'top_payment_country_bahamas', 'top_payment_country_bahrain', 'top_payment_country_bangladesh', 'top_payment_country_barbados', 'top_payment_country_belarus', 'top_payment_country_belgium', 'top_payment_country_belize', 'top_payment_country_benin', 'top_payment_country_bermuda', 'top_payment_country_bhutan', 'top_payment_country_bolivia', 'top_payment_country_bonaire, sint eustatius, and saba', 'top_payment_country_bosnia and herzegovina', 'top_payment_country_botswana', 'top_payment_country_brazil', 'top_payment_country_british indian ocean territory', 'top_payment_country_british virgin islands', 'top_payment_country_brunei', 'top_payment_country_bulgaria', 'top_payment_country_burkina faso', 'top_payment_country_cabo verde', 'top_payment_country_cambodia', 'top_payment_country_cameroon', 'top_payment_country_canada', 'top_payment_country_cayman islands', 'top_payment_country_chad', 'top_payment_country_chile', 'top_payment_country_china', 'top_payment_country_colombia', 'top_payment_country_comoros', 'top_payment_country_cook islands', 'top_payment_country_costa rica', 'top_payment_country_croatia', 'top_payment_country_curaçao', 'top_payment_country_cyprus', 'top_payment_country_czechia', 'top_payment_country_denmark', 'top_payment_country_djibouti', 'top_payment_country_dominica', 'top_payment_country_dominican republic', 'top_payment_country_ecuador', 'top_payment_country_egypt', 'top_payment_country_el salvador', 'top_payment_country_equatorial guinea', 'top_payment_country_estonia', 'top_payment_country_eswatini', 'top_payment_country_falkland islands', 'top_payment_country_faroe islands', 'top_payment_country_federated states of micronesia', 'top_payment_country_fiji', 'top_payment_country_finland', 'top_payment_country_france', 'top_payment_country_french guiana', 'top_payment_country_french polynesia', 'top_payment_country_gabon', 'top_payment_country_gambia', 'top_payment_country_georgia', 'top_payment_country_germany', 'top_payment_country_ghana', 'top_payment_country_gibraltar', 'top_payment_country_greece', 'top_payment_country_greenland', 'top_payment_country_grenada', 'top_payment_country_guadeloupe', 'top_payment_country_guam', 'top_payment_country_guatemala', 'top_payment_country_guernsey', 'top_payment_country_guinea', 'top_payment_country_guyana', 'top_payment_country_haiti', 'top_payment_country_honduras', 'top_payment_country_hong kong', 'top_payment_country_hungary', 'top_payment_country_iceland', 'top_payment_country_india', 'top_payment_country_indonesia', 'top_payment_country_iraq', 'top_payment_country_ireland', 'top_payment_country_isle of man', 'top_payment_country_israel', 'top_payment_country_italy', 'top_payment_country_ivory coast', 'top_payment_country_jamaica', 'top_payment_country_japan', 'top_payment_country_jersey', 'top_payment_country_jordan', 'top_payment_country_kazakhstan', 'top_payment_country_kenya', 'top_payment_country_kiribati', 'top_payment_country_kosovo', 'top_payment_country_kuwait', 'top_payment_country_kyrgyzstan', 'top_payment_country_laos', 'top_payment_country_latvia', 'top_payment_country_lesotho', 'top_payment_country_liechtenstein', 'top_payment_country_lithuania', 'top_payment_country_luxembourg', 'top_payment_country_macao', 'top_payment_country_madagascar', 'top_payment_country_malawi', 'top_payment_country_malaysia', 'top_payment_country_maldives', 'top_payment_country_malta', 'top_payment_country_marshall islands', 'top_payment_country_martinique', 'top_payment_country_mauritania', 'top_payment_country_mauritius', 'top_payment_country_mayotte', 'top_payment_country_mexico', 'top_payment_country_missing', 'top_payment_country_moldova', 'top_payment_country_monaco', 'top_payment_country_mongolia', 'top_payment_country_montenegro', 'top_payment_country_morocco', 'top_payment_country_mozambique', 'top_payment_country_namibia', 'top_payment_country_nauru', 'top_payment_country_nepal', 'top_payment_country_new caledonia', 'top_payment_country_new zealand', 'top_payment_country_niger', 'top_payment_country_nigeria', 'top_payment_country_north macedonia', 'top_payment_country_northern mariana islands', 'top_payment_country_norway', 'top_payment_country_oman', 'top_payment_country_pakistan', 'top_payment_country_palau', 'top_payment_country_palestine', 'top_payment_country_panama', 'top_payment_country_papua new guinea', 'top_payment_country_paraguay', 'top_payment_country_peru', 'top_payment_country_philippines', 'top_payment_country_poland', 'top_payment_country_portugal', 'top_payment_country_puerto rico', 'top_payment_country_qatar', 'top_payment_country_romania', 'top_payment_country_rwanda', 'top_payment_country_réunion', 'top_payment_country_saint lucia', 'top_payment_country_saint martin', 'top_payment_country_samoa', 'top_payment_country_san marino', 'top_payment_country_saudi arabia', 'top_payment_country_senegal', 'top_payment_country_serbia', 'top_payment_country_seychelles', 'top_payment_country_sierra leone', 'top_payment_country_singapore', 'top_payment_country_sint maarten', 'top_payment_country_slovakia', 'top_payment_country_slovenia', 'top_payment_country_solomon islands', 'top_payment_country_south africa', 'top_payment_country_south korea', 'top_payment_country_spain', 'top_payment_country_sri lanka', 'top_payment_country_st kitts and nevis', 'top_payment_country_st vincent and grenadines', 'top_payment_country_suriname', 'top_payment_country_sweden', 'top_payment_country_switzerland', 'top_payment_country_são tomé and príncipe', 'top_payment_country_taiwan', 'top_payment_country_tajikistan', 'top_payment_country_tanzania', 'top_payment_country_thailand', 'top_payment_country_the netherlands', 'top_payment_country_timor-leste', 'top_payment_country_togo', 'top_payment_country_tonga', 'top_payment_country_trinidad and tobago', 'top_payment_country_turks and caicos islands', 'top_payment_country_türkiye', 'top_payment_country_u.s. virgin islands', 'top_payment_country_uganda', 'top_payment_country_ukraine', 'top_payment_country_united arab emirates', 'top_payment_country_united kingdom', 'top_payment_country_united states', 'top_payment_country_uruguay', 'top_payment_country_uzbekistan', 'top_payment_country_vanuatu', 'top_payment_country_vietnam', 'top_payment_country_zambia', 'top_payment_country_zimbabwe', 'top_payment_country_åland islands', 'top_card_country_afghanistan', 'top_card_country_albania', 'top_card_country_algeria', 'top_card_country_american samoa', 'top_card_country_andorra', 'top_card_country_angola', 'top_card_country_anguilla', 'top_card_country_antigua and barbuda', 'top_card_country_argentina', 'top_card_country_armenia', 'top_card_country_aruba', 'top_card_country_australia', 'top_card_country_austria', 'top_card_country_azerbaijan', 'top_card_country_bahamas', 'top_card_country_bahrain', 'top_card_country_bangladesh', 'top_card_country_barbados', 'top_card_country_belarus', 'top_card_country_belgium', 'top_card_country_belize', 'top_card_country_benin', 'top_card_country_bermuda', 'top_card_country_bhutan', 'top_card_country_bolivia', 'top_card_country_bonaire, sint eustatius, and saba', 'top_card_country_bosnia and herzegovina', 'top_card_country_botswana', 'top_card_country_brazil', 'top_card_country_british virgin islands', 'top_card_country_brunei', 'top_card_country_bulgaria', 'top_card_country_burkina faso', 'top_card_country_burundi', 'top_card_country_cambodia', 'top_card_country_cameroon', 'top_card_country_canada', 'top_card_country_cape verde', 'top_card_country_cayman islands', 'top_card_country_central african republic', 'top_card_country_chad', 'top_card_country_chile', 'top_card_country_china', 'top_card_country_colombia', 'top_card_country_comoros', 'top_card_country_cook islands', 'top_card_country_costa rica', 'top_card_country_croatia', 'top_card_country_curacao', 'top_card_country_cyprus', 'top_card_country_czech republic', 'top_card_country_democratic republic of the congo', 'top_card_country_denmark', 'top_card_country_djibouti', 'top_card_country_dominica', 'top_card_country_dominican republic', 'top_card_country_east timor', 'top_card_country_ecuador', 'top_card_country_egypt', 'top_card_country_el salvador', 'top_card_country_equatorial guinea', 'top_card_country_estonia', 'top_card_country_ethiopia', 'top_card_country_fiji', 'top_card_country_finland', 'top_card_country_france', 'top_card_country_french polynesia', 'top_card_country_gabon', 'top_card_country_gambia', 'top_card_country_georgia', 'top_card_country_germany', 'top_card_country_ghana', 'top_card_country_gibraltar', 'top_card_country_greece', 'top_card_country_grenada', 'top_card_country_guam', 'top_card_country_guatemala', 'top_card_country_guinea', 'top_card_country_guyana', 'top_card_country_haiti', 'top_card_country_honduras', 'top_card_country_hong kong', 'top_card_country_hungary', 'top_card_country_iceland', 'top_card_country_india', 'top_card_country_indonesia', 'top_card_country_iran', 'top_card_country_iraq', 'top_card_country_ireland', 'top_card_country_israel', 'top_card_country_italy', 'top_card_country_ivory coast', 'top_card_country_jamaica', 'top_card_country_japan', 'top_card_country_jordan', 'top_card_country_kazakhstan', 'top_card_country_kenya', 'top_card_country_kiribati', 'top_card_country_kosovo', 'top_card_country_kuwait', 'top_card_country_kyrgyzstan', 'top_card_country_laos', 'top_card_country_latvia', 'top_card_country_lebanon', 'top_card_country_lesotho', 'top_card_country_liberia', 'top_card_country_libya', 'top_card_country_liechtenstein', 'top_card_country_lithuania', 'top_card_country_luxembourg', 'top_card_country_macau', 'top_card_country_macedonia', 'top_card_country_madagascar', 'top_card_country_malawi', 'top_card_country_malaysia', 'top_card_country_maldives', 'top_card_country_mali', 'top_card_country_malta', 'top_card_country_marocco', 'top_card_country_mauritania', 'top_card_country_mauritius', 'top_card_country_mexico', 'top_card_country_micronesia', 'top_card_country_missing', 'top_card_country_moldova', 'top_card_country_mongolia', 'top_card_country_montenegro', 'top_card_country_mozambique', 'top_card_country_myanmar', 'top_card_country_namibia', 'top_card_country_nepal', 'top_card_country_netherlands', 'top_card_country_new caledonia', 'top_card_country_new zealand', 'top_card_country_nicaragua', 'top_card_country_niger', 'top_card_country_nigeria', 'top_card_country_northern mariana islands', 'top_card_country_norway', 'top_card_country_oman', 'top_card_country_pakistan', 'top_card_country_palestine', 'top_card_country_panama', 'top_card_country_papua new guinea', 'top_card_country_paraguay', 'top_card_country_peru', 'top_card_country_philippines', 'top_card_country_poland', 'top_card_country_portugal', 'top_card_country_puerto rico', 'top_card_country_qatar', 'top_card_country_republic of the congo', 'top_card_country_romania', 'top_card_country_rwanda', 'top_card_country_saint kitts and nevis', 'top_card_country_saint lucia', 'top_card_country_saint vincent and the grenadines', 'top_card_country_samoa', 'top_card_country_san marino', 'top_card_country_saudi arabia', 'top_card_country_senegal', 'top_card_country_serbia', 'top_card_country_seychelles', 'top_card_country_sierra leone', 'top_card_country_singapore', 'top_card_country_sint maarten', 'top_card_country_slovakia', 'top_card_country_slovenia', 'top_card_country_solomon islands', 'top_card_country_somalia', 'top_card_country_south africa', 'top_card_country_south korea', 'top_card_country_south sudan', 'top_card_country_spain', 'top_card_country_sri lanka', 'top_card_country_sudan', 'top_card_country_suriname', 'top_card_country_swaziland', 'top_card_country_sweden', 'top_card_country_switzerland', 'top_card_country_taiwan', 'top_card_country_tajikistan', 'top_card_country_tanzania', 'top_card_country_thailand', 'top_card_country_togo', 'top_card_country_tonga', 'top_card_country_trinidad and tobago', 'top_card_country_tunisia', 'top_card_country_turkey', 'top_card_country_turkmenistan', 'top_card_country_turks and caicos islands', 'top_card_country_u.s. virgin islands', 'top_card_country_uganda', 'top_card_country_ukraine', 'top_card_country_united arab emirates', 'top_card_country_united kingdom', 'top_card_country_united states', 'top_card_country_uruguay', 'top_card_country_uzbekistan', 'top_card_country_vanuatu', 'top_card_country_venezuela', 'top_card_country_vietnam', 'top_card_country_yemen', 'top_card_country_zambia', 'top_card_country_zimbabwe', 'traffic_type_cpa', 'traffic_type_organic', 'traffic_type_ppc', 'traffic_type_remarketing', 'traffic_type_unknown', 'gender_female', 'gender_male', 'reg_country_afghanistan', 'reg_country_albania', 'reg_country_algeria', 'reg_country_american samoa', 'reg_country_andorra', 'reg_country_angola', 'reg_country_anguilla', 'reg_country_antarctica', 'reg_country_antigua and barbuda', 'reg_country_argentina', 'reg_country_armenia', 'reg_country_aruba', 'reg_country_australia', 'reg_country_austria', 'reg_country_azerbaijan', 'reg_country_bahamas', 'reg_country_bahrain', 'reg_country_bangladesh', 'reg_country_barbados', 'reg_country_belgium', 'reg_country_belize', 'reg_country_benin', 'reg_country_bermuda', 'reg_country_bhutan', 'reg_country_bolivia', 'reg_country_bonaire, sint eustatius, and saba', 'reg_country_botswana', 'reg_country_brazil', 'reg_country_british indian ocean territory', 'reg_country_british virgin islands', 'reg_country_brunei', 'reg_country_bulgaria', 'reg_country_burkina faso', 'reg_country_cambodia', 'reg_country_cameroon', 'reg_country_canada', 'reg_country_cape verde', 'reg_country_cayman islands', 'reg_country_chad', 'reg_country_chile', 'reg_country_china', 'reg_country_christmas island', 'reg_country_colombia', 'reg_country_comoros', 'reg_country_cook islands', 'reg_country_costa rica', 'reg_country_croatia', 'reg_country_curacao', 'reg_country_cyprus', 'reg_country_czech republic', 'reg_country_denmark', 'reg_country_djibouti', 'reg_country_dominica', 'reg_country_dominican republic', 'reg_country_east timor', 'reg_country_ecuador', 'reg_country_egypt', 'reg_country_el salvador', 'reg_country_equatorial guinea', 'reg_country_eritrea', 'reg_country_estonia', 'reg_country_ethiopia', 'reg_country_falkland islands', 'reg_country_faroe islands', 'reg_country_fiji', 'reg_country_finland', 'reg_country_france', 'reg_country_french polynesia', 'reg_country_gabon', 'reg_country_gambia', 'reg_country_georgia', 'reg_country_germany', 'reg_country_ghana', 'reg_country_gibraltar', 'reg_country_greece', 'reg_country_greenland', 'reg_country_grenada', 'reg_country_guadeloupe', 'reg_country_guam', 'reg_country_guatemala', 'reg_country_guernsey', 'reg_country_guinea', 'reg_country_guyana', 'reg_country_guyane', 'reg_country_haiti', 'reg_country_honduras', 'reg_country_hong kong', 'reg_country_hungary', 'reg_country_iceland', 'reg_country_india', 'reg_country_indonesia', 'reg_country_ireland', 'reg_country_isle of man', 'reg_country_israel', 'reg_country_italy', 'reg_country_ivory coast', 'reg_country_jamaica', 'reg_country_japan', 'reg_country_jersey', 'reg_country_jordan', 'reg_country_kazakhstan', 'reg_country_kenya', 'reg_country_kiribati', 'reg_country_kosovo', 'reg_country_kuwait', 'reg_country_kyrgyzstan', 'reg_country_laos', 'reg_country_latvia', 'reg_country_lesotho', 'reg_country_liechtenstein', 'reg_country_lithuania', 'reg_country_luxembourg', 'reg_country_macau', 'reg_country_macedonia', 'reg_country_madagascar', 'reg_country_malawi', 'reg_country_malaysia', 'reg_country_maldives', 'reg_country_malta', 'reg_country_marocco', 'reg_country_marshall islands', 'reg_country_martinique', 'reg_country_mauritania', 'reg_country_mauritius', 'reg_country_mayotte', 'reg_country_mexico', 'reg_country_micronesia', 'reg_country_missing', 'reg_country_moldova', 'reg_country_monaco', 'reg_country_mongolia', 'reg_country_montenegro', 'reg_country_mozambique', 'reg_country_namibia', 'reg_country_nauru', 'reg_country_nepal', 'reg_country_netherlands', 'reg_country_new caledonia', 'reg_country_new zealand', 'reg_country_niger', 'reg_country_nigeria', 'reg_country_norfolk island', 'reg_country_northern mariana islands', 'reg_country_norway', 'reg_country_oman', 'reg_country_pakistan', 'reg_country_palau', 'reg_country_palestine', 'reg_country_panama', 'reg_country_papua new guinea', 'reg_country_paraguay', 'reg_country_peru', 'reg_country_philippines', 'reg_country_pitcairn', 'reg_country_poland', 'reg_country_portugal', 'reg_country_puerto rico', 'reg_country_qatar', 'reg_country_reunion', 'reg_country_romania', 'reg_country_rwanda', 'reg_country_saint barthelemy', 'reg_country_saint kitts and nevis', 'reg_country_saint lucia', 'reg_country_saint martin', 'reg_country_saint pierre and miquelon', 'reg_country_saint vincent and the grenadines', 'reg_country_samoa', 'reg_country_san marino', 'reg_country_sao tome and principe', 'reg_country_saudi arabia', 'reg_country_senegal', 'reg_country_serbia', 'reg_country_seychelles', 'reg_country_sierra leone', 'reg_country_singapore', 'reg_country_sint maarten', 'reg_country_slovakia', 'reg_country_slovenia', 'reg_country_solomon islands', 'reg_country_south africa', 'reg_country_south korea', 'reg_country_spain', 'reg_country_sri lanka', 'reg_country_suriname', 'reg_country_swaziland', 'reg_country_sweden', 'reg_country_switzerland', 'reg_country_taiwan', 'reg_country_tajikistan', 'reg_country_tanzania', 'reg_country_thailand', 'reg_country_the french southern and antarctic lands', 'reg_country_togo', 'reg_country_tonga', 'reg_country_trinidad and tobago', 'reg_country_turkey', 'reg_country_turkmenistan', 'reg_country_turks and caicos islands', 'reg_country_tuvalu', 'reg_country_u.s. virgin islands', 'reg_country_uganda', 'reg_country_ukraine', 'reg_country_united arab emirates', 'reg_country_united kingdom', 'reg_country_united states', 'reg_country_uruguay', 'reg_country_uzbekistan', 'reg_country_vanuatu', 'reg_country_vatican', 'reg_country_venezuela', 'reg_country_vietnam', 'reg_country_wallis and futuna', 'reg_country_zambia', 'reg_country_zimbabwe', 'reg_country_åland islands', 'email_domain_gmail.com', 'email_domain_hotmail.com', 'email_domain_missing', 'email_domain_outlook.com', 'email_domain_yahoo.com']


# make sure columns match train
X_test = X_test.reindex(columns=train_feature_cols, fill_value=0)

# numeric cleanup
for col in X_test.columns:
    X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

X_test = X_test.replace([float("inf"), float("-inf")], pd.NA).fillna(-1)

# load model
model = XGBClassifier()
model.load_model("xgb_model.json")

# predict
test_proba = model.predict_proba(X_test)[:, 1]
test_pred = (test_proba >= 0.9).astype(int)

# -----------------------------
# 1. save submission: user_id|is_fraud
# -----------------------------
submission = pd.DataFrame({
    "id_user": test_ids,
    "is_fraud": test_pred,
})

submission.to_csv("submission_pipe.csv", index=False)

# -----------------------------
# 2. attach predictions to full test dataset
# -----------------------------
test_dataset["fraud_proba"] = test_proba
test_dataset["is_fraud"] = test_pred

# keep only predicted fraud users
fraud_users = test_dataset[test_dataset["is_fraud"] == 1].copy()

print("Predicted fraud users:", len(fraud_users))

# -----------------------------
# 3. split all fraud users with all features into txt files by 80 users
# -----------------------------
os.makedirs("fraud_txt", exist_ok=True)

chunk_size = 80
unique_fraud_ids = fraud_users["id_user"].drop_duplicates().tolist()

cols_keep = [
    "transaction_number",
    "fail_count",
    "success_count",
    "antifraud_count",
    "fail_rate",
    "fraud_error_rate",
    "unique_card_mask_hash_count",
    "unique_card_holder_norm_count",
    "country_contradiction_score",
    "geo_card_fail_interaction",
    "geo_holder_interaction",
    "fs_fail_rate",
    "fs_fraud_error_rate",
    "last10_fail_rate",
    "last10_fraud_error_rate",
    "last10_unique_cards",
    "last20_fail_rate",
    "last20_fraud_error_rate",
    "last20_unique_cards",
    "last10_burst_pressure",
    "last20_burst_pressure",
    "max_fails_in_1h",
    "max_unique_cards_in_1h"
]

for i in range(0, len(unique_fraud_ids), chunk_size):
    part_ids = unique_fraud_ids[i:i + chunk_size]
    file_num = i // chunk_size + 1
    file_path = f"fraud_txt/fraud_users_{file_num}.txt"

    with open(file_path, "w", encoding="utf-8") as f:
        for user_id in part_ids:
            user_part = fraud_users.loc[fraud_users["id_user"] == user_id, cols_keep]

            if user_part.empty:
                continue

            row = user_part.iloc[0].to_dict()

            # optional: remove empty / useless values
            row = {
                k: v for k, v in row.items()
                if v is not None and str(v) not in {"nan", "NaN", "", "-1"}
            }

            obj = {
                "id_user": str(user_id),
                **row
            }

            f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")))
            f.write("\n")

print("Done")