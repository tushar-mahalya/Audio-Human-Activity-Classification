-- create the databases
CREATE DATABASE IF NOT EXISTS `audio`;

USE `audio`;

-- create tables
CREATE TABLE IF NOT EXISTS `input` (
    input_id INT AUTO_INCREMENT PRIMARY KEY,
    label VARCHAR(255) NOT NULL,
    mfcc BLOB,
    input_timestamp VARCHAR(255) NOT NULL
);

CREATE TABLE IF NOT EXISTS `labels` (
    label_id INT AUTO_INCREMENT PRIMARY KEY,
    input_label VARCHAR(255) NOT NULL,
    classified_label VARCHAR(255) NOT NULL,
    classified_label_probability VARCHAR(255) NOT NULL,
    classified_all VARCHAR(512) NOT NULL,
    classified_timestamp VARCHAR(255) NOT NULL
);

CREATE TABLE IF NOT EXISTS `predictions` (
    prediction_id INT AUTO_INCREMENT PRIMARY KEY,
    label_id INT NOT NULL,
    classified_label VARCHAR(255) NOT NULL,
    classified_label_probability VARCHAR(255) NOT NULL,
    next_location_label VARCHAR(255) NOT NULL,
    next_location_label_probability VARCHAR(255) NOT NULL,
    prediction_timestamp VARCHAR(255) NOT NULL
);
