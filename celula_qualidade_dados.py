#!/usr/bin/env python3

import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, mean, stddev, count, avg, sum, monotonically_increasing_id
from pyspark.sql.types import ArrayType, StructType, StructField, StringType, IntegerType, FloatType, DecimalType
import requests
import psycopg2
from psycopg2 import Error
from datetime import datetime
from decimal import Decimal
import json

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_spark():
    """Inicializa a SparkSession."""
    try:
        spark = SparkSession.builder \
            .appName("Qualidade_de_dados") \
            .config("spark.jars.packages", "org.postgresql:postgresql:42.7.4") \
            .getOrCreate()
        logger.info("SparkSession inicializada com sucesso.")
        return spark
    except Exception as e:
        logger.error(f"Erro ao inicializar SparkSession: {str(e)}")
        raise

def extract_data(url_api, headers=None, params=None):
    """Extrai dados da API fornecida."""
    try:
        response = requests.get(url_api, headers=headers, params=params)
        response.raise_for_status()
        logger.info(f"Resposta da API recebida com sucesso: {url_api}")
        return response.text
    except requests.exceptions.RequestException as e:
        logger.error(f"Erro ao acessar API {url_api}: {str(e)}")
        return None

def define_schema():
    """Define o schema para o DataFrame."""
    rating = StructType([
        StructField("rate", DecimalType(10, 2), True),
        StructField("count", IntegerType(), True),
    ])
    return StructType([
        StructField("id", IntegerType(), True),
        StructField("title", StringType(), True),
        StructField("price", DecimalType(10, 2), True),
        StructField("description", StringType(), True),
        StructField("category", StringType(), True),
        StructField("image", StringType(), True),
        StructField("rating", rating, True)
    ])

def process_data(spark, data, schema):
    """Processa os dados da API para um DataFrame limpo."""
    try:
        # Normalizar valores para Decimal
        for item in data:
            item["price"] = Decimal(str(item["price"]))
            item["rating"]["rate"] = Decimal(str(item["rating"]["rate"]))
        
        # Criar DataFrame
        df = spark.createDataFrame(data, schema=schema)
        logger.info("DataFrame criado com sucesso.")

        # Limpeza de dados
        df_clean = df.dropna(subset=["price", "category", "rating.rate"])
        df_clean = df_clean.withColumn("rate", col("rating.rate"))
        df_clean = df_clean.filter((col("price") >= 0) & (col("rate").between(0, 5)))

        # Remover outliers com método IQR
        quantiles_price = df_clean.approxQuantile("price", [0.25, 0.75], 0.05)
        quantiles_rate = df_clean.approxQuantile("rate", [0.25, 0.75], 0.05)

        q1_price, q3_price = quantiles_price[0], quantiles_price[1]
        iqr_price = q3_price - q1_price
        lower_bound_price = q1_price - 1.5 * iqr_price
        upper_bound_price = q3_price + 1.5 * iqr_price

        q1_rate, q3_rate = quantiles_rate[0], quantiles_rate[1]
        iqr_rate = q3_rate - q1_rate
        lower_bound_rate = q1_rate - 1.5 * iqr_rate
        upper_bound_rate = q3_rate + 1.5 * iqr_rate

        df_clean = df_clean.filter(
            (col("price") >= lower_bound_price) & (col("price") <= upper_bound_price) &
            (col("rate") >= lower_bound_rate) & (col("rate") <= upper_bound_rate)
        )

        # Filtros adicionais
        df_clean = df_clean.filter((col("price") >= 100) & (col("rate") >= 3.5))
        logger.info("Dados limpos com sucesso.")

        # Sumarização por categoria
        df_summary = df_clean.groupBy("category").agg(
            avg("price").alias("preco_medio"),
            avg("rate").alias("avaliacao_media")
        ).withColumnRenamed("category", "categoria")
        logger.info("Sumarização por categoria concluída.")

        return df_clean, df_summary
    except Exception as e:
        logger.error(f"Erro ao processar dados: {str(e)}")
        raise

def save_to_postgres(df, table_name, db_properties, mode="append"):
    """Salva o DataFrame no PostgreSQL."""
    try:
        df.write \
            .format("jdbc") \
            .option("url", db_properties["url"]) \
            .option("dbtable", table_name) \
            .option("driver", db_properties["driver"]) \
            .option("user", db_properties["user"]) \
            .option("password", db_properties["password"]) \
            .mode(mode) \
            .save()
        logger.info(f"DataFrame salvo na tabela {table_name} com sucesso.")
    except Exception as e:
        logger.error(f"Erro ao salvar DataFrame na tabela {table_name}: {str(e)}")
        raise

def create_tables(conn):
    """Cria as tabelas no PostgreSQL se não existirem."""
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS produtos (
                id INTEGER,
                title TEXT,
                price FLOAT,
                description TEXT,
                category TEXT,
                image TEXT,
                rating_rate FLOAT,
                rating_count INTEGER
            );
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS categoria_media (
                category TEXT,
                mean_price FLOAT,
                mean_rate FLOAT
            );
        """)
        conn.commit()
        logger.info("Tabelas criadas com sucesso.")
    except Exception as e:
        logger.error(f"Erro ao criar tabelas: {str(e)}")
        conn.rollback()
        raise
    finally:
        cursor.close()

def main():
    """Função principal para executar o pipeline."""
    # Configurações
    url_api = "https://fakestoreapi.com/products"
    db_properties = {
        "url": "jdbc:postgresql://localhost:5432/desafio_tecnico",
        "driver": "org.postgresql.Driver",
        "user": "desafio_tecnico_user",
        "password": "desafiotecnico123",
        "dbtable_produtos": "produtos",
        "dbtable_summary": "categoria_media"
    }
    db_config = {
        "host": "localhost",
        "port": "5432",
        "database": "desafio_tecnico",
        "user": "desafio_tecnico_user",
        "password": "desafiotecnico123"
    }

    # Inicializar Spark
    spark = initialize_spark()

    try:
        # Extrair dados da API
        data_api = extract_data(url_api)
        if not data_api:
            raise ValueError("Falha ao obter dados da API.")
        
        data = json.loads(data_api)
        schema = define_schema()
        df_clean, df_summary = process_data(spark, data, schema)

        # Conectar ao PostgreSQL
        conn = psycopg2.connect(
            host=db_config["host"],
            port=db_config["port"],
            database=db_config["database"],
            user=db_config["user"],
            password=db_config["password"]
        )
        conn.autocommit = False
        create_tables(conn)

        # Salvar em batches
        batch_size = 5
        total_rows = df_clean.count()
        num_batches = (total_rows + batch_size - 1) // batch_size  # Correção no cálculo

        df_clean_with_index = df_clean.withColumn("row_index", monotonically_increasing_id())
        for i in range(num_batches):
            start_index = i * batch_size
            end_index = start_index + batch_size
            df_batch = df_clean_with_index.filter(
                (col("row_index") >= start_index) & (col("row_index") < end_index)
            ).drop("row_index")

            df_batch = df_batch.select(
                col("id"),
                col("title"),
                col("price"),
                col("description"),
                col("category"),
                col("image"),
                col("rating.rate").alias("rating_rate"),
                col("rating.count").alias("rating_count")
            )

            logger.info(f"Salvando batch {i+1} com {df_batch.count()} registros na tabela 'produtos'...")
            save_to_postgres(df_batch, db_properties["dbtable_produtos"], db_properties, mode="append")

        # Salvar resumo
        logger.info("Salvando df_summary na tabela 'categoria_media'...")
        save_to_postgres(df_summary, db_properties["dbtable_summary"], db_properties, mode="overwrite")

        conn.commit()
        logger.info("Transação confirmada com sucesso.")

    except Exception as e:
        logger.error(f"Erro durante a execução do pipeline: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
            logger.info("Transação revertida devido a erro.")
        raise
    finally:
        if 'conn' in locals():
            conn.close()
        spark.stop()
        logger.info("SparkSession e conexão com banco fechadas.")

if __name__ == '__main__':
    main()