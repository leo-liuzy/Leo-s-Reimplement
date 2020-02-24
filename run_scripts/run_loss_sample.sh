#!/bin/bash -i

if [ "$1" == "0" ]; then
    TASKS="yelp_review_full_csv ag_news_csv dbpedia_csv amazon_review_full_csv yahoo_answers_csv"
elif [ "$1" == "1" ]; then
    TASKS="dbpedia_csv yahoo_answers_csv ag_news_csv amazon_review_full_csv yelp_review_full_csv"
elif [ "$1" == "2" ]; then
    TASKS="yelp_review_full_csv yahoo_answers_csv amazon_review_full_csv dbpedia_csv ag_news_csv"
elif [ "$1" == "3" ]; then
    TASKS="ag_news_csv yelp_review_full_csv amazon_review_full_csv yahoo_answers_csv dbpedia_csv"
elif [ "$1" == "4" ]; then
    TASKS="amazon_review_full_csv ag_news_csv yelp_review_full_csv yahoo_answers_csv dbpedia_csv"
fi

export DIR="/home/leo/episodic_memory"

python3 "$DIR/train/train_class_loss_sample.py" --tasks $TASKS --output_dir "output$2" --sampler_choice random --batch_size 8

# python test_modified.py --output_dir "output$2" --adapt_steps 30 --n_test 200 --logging_steps 20 --test_log_filename log_test_adam.txt --adapt_lr 1e-3
# python test_modified.py --output_dir "output$2" --adapt_steps 30 --n_test 200 --logging_steps 20 --test_log_filename log_test_adam.txt --adapt_lr 5e-3
