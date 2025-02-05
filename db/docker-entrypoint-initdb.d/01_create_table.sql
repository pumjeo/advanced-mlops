CREATE DATABASE IF NOT EXISTS mlops;

USE mlops;

DROP TABLE IF EXISTS `mlops`.`customer_info`;
CREATE TABLE `mlops`.`customer_info` (
  `id` varchar(10) NOT NULL,
  `customer_id` varchar(10) NOT NULL,
  `date` varchar(10) NOT NULL,
  `name` varchar(50) DEFAULT NULL,
  `ssn` bigint DEFAULT NULL,
  PRIMARY KEY (`id`,`customer_id`, `date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
PARTITION BY KEY(`date`)
PARTITIONS 8
;

DROP TABLE IF EXISTS `mlops`.`credit_score`;
CREATE TABLE `mlops`.`credit_score` (
  `id` varchar(10) NOT NULL,
  `customer_id` varchar(10) NOT NULL,
  `date` varchar(10) NOT NULL,
  `credit_score` varchar(10) NOT NULL,
  PRIMARY KEY (`id`,`customer_id`, `date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
PARTITION BY KEY(`date`)
PARTITIONS 8
;

DROP TABLE IF EXISTS `mlops`.`credit_score_features`;
CREATE TABLE `mlops`.`credit_score_features` (
  `id` varchar(10) NOT NULL,
  `customer_id` varchar(10) NOT NULL,
  `date` varchar(10) NOT NULL,
  `name` varchar(50) DEFAULT NULL,
  `age` float DEFAULT NULL,
  `ssn` bigint DEFAULT NULL,
  `occupation` varchar(50) DEFAULT NULL,
  `annual_income` float DEFAULT NULL,
  `monthly_inhand_salary` float DEFAULT NULL,
  `num_bank_accounts` float DEFAULT NULL,
  `num_credit_card` float DEFAULT NULL,
  `interest_rate` float DEFAULT NULL,
  `num_of_loan` float DEFAULT NULL,
  `type_of_loan` text DEFAULT NULL,
  `delay_from_due_date` float DEFAULT NULL,
  `num_of_delayed_payment` float DEFAULT NULL,
  `changed_credit_limit` float DEFAULT NULL,
  `num_credit_inquiries` float DEFAULT NULL,
  `credit_mix` varchar(20) DEFAULT NULL,
  `outstanding_debt` float DEFAULT NULL,
  `credit_utilization_ratio` float DEFAULT NULL,
  `credit_history_age` float DEFAULT NULL,
  `payment_of_min_amount` varchar(10) DEFAULT NULL,
  `total_emi_per_month` float DEFAULT NULL,
  `amount_invested_monthly` float DEFAULT NULL,
  `payment_behaviour` varchar(50) DEFAULT NULL,
  `monthly_balance` float DEFAULT NULL,
  PRIMARY KEY (`id`,`customer_id`, `date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
PARTITION BY KEY(`date`)
PARTITIONS 8
;

DROP TABLE IF EXISTS `mlops`.`credit_score_features_target`;
CREATE TABLE `mlops`.`credit_score_features_target` (
  `base_dt` varchar(10) NOT NULL,
  `id` varchar(10) NOT NULL,
  `customer_id` varchar(10) NOT NULL,
  `date` varchar(10) NOT NULL,
  `age` float DEFAULT NULL,
  `occupation` varchar(50) DEFAULT NULL,
  `annual_income` float DEFAULT NULL,
  `monthly_inhand_salary` float DEFAULT NULL,
  `num_bank_accounts` float DEFAULT NULL,
  `num_credit_card` float DEFAULT NULL,
  `interest_rate` float DEFAULT NULL,
  `num_of_loan` float DEFAULT NULL,
  `type_of_loan` text DEFAULT NULL,
  `delay_from_due_date` float DEFAULT NULL,
  `num_of_delayed_payment` float DEFAULT NULL,
  `changed_credit_limit` float DEFAULT NULL,
  `num_credit_inquiries` float DEFAULT NULL,
  `credit_mix` varchar(20) DEFAULT NULL,
  `outstanding_debt` float DEFAULT NULL,
  `credit_utilization_ratio` float DEFAULT NULL,
  `credit_history_age` float DEFAULT NULL,
  `payment_of_min_amount` varchar(10) DEFAULT NULL,
  `total_emi_per_month` float DEFAULT NULL,
  `amount_invested_monthly` float DEFAULT NULL,
  `payment_behaviour` varchar(50) DEFAULT NULL,
  `monthly_balance` float DEFAULT NULL,
  `credit_score` varchar(10) NOT NULL,
  PRIMARY KEY (`id`,`customer_id`, `date`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
PARTITION BY KEY(`date`)
PARTITIONS 8
;
