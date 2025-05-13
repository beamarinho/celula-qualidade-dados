import unittest
from unittest.mock import patch, MagicMock
import json
from decimal import Decimal
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DecimalType
from pyspark.sql.functions import col
import requests
import pytest

# Importar funções do código original (assumindo que estão em um módulo chamado `qualidade_dados.py`)
from qualidade_dados import extract_data, define_schema, save_to_postgres

# Dados de teste simulados
mock_api_response = [
    {
        "id": 1,
        "title": "Produto 1",
        "price": 150.00,
        "description": "Descrição do Produto 1",
        "category": "electronics",
        "image": "http://image1.jpg",
        "rating": {"rate": 4.5, "count": 100}
    },
    {
        "id": 2,
        "title": "Produto 2",
        "price": 80.00,
        "description": "Descrição do Produto 2",
        "category": "electronics",
        "image": "http://image2.jpg",
        "rating": {"rate": 3.0, "count": 50}
    },
    {
        "id": 3,
        "title": "Produto 3",
        "price": 200.00,
        "description": None,
        "category": "clothing",
        "image": "http://image3.jpg",
        "rating": {"rate": 4.0, "count": 75}
    }
]

class TestQualidadeDados(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Inicializar SparkSession para testes
        cls.spark = SparkSession.builder \
            .appName("TestQualidadeDados") \
            .master("local[2]") \
            .getOrCreate()

    @classmethod
    def tearDownClass(cls):
        # Finalizar SparkSession
        cls.spark.stop()

    @patch('requests.get')
    def test_extract_data_success(self, mock_get):
        # Simular resposta bem-sucedida da API
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_api_response
        mock_response.text = json.dumps(mock_api_response)
        mock_get.return_value = mock_response

        result = extract_data("http://fakeapi.com")
        self.assertEqual(result, json.dumps(mock_api_response))
        mock_get.assert_called_once_with("http://fakeapi.com", headers=None, params=None)

    @patch('requests.get')
    def test_extract_data_failure(self, mock_get):
        # Simular falha na API
        mock_get.side_effect = requests.exceptions.RequestException("Erro de conexão")
        result = extract_data("http://fakeapi.com")
        self.assertIsNone(result)

    def test_define_schema(self):
        schema = define_schema()
        self.assertIsInstance(schema, StructType)
        self.assertEqual(len(schema.fields), 7)
        self.assertEqual(schema.fields[0].name, "id")
        self.assertEqual(schema.fields[0].dataType, IntegerType())
        self.assertEqual(schema.fields[2].name, "price")
        self.assertEqual(schema.fields[2].dataType, DecimalType(10, 2))
        self.assertEqual(schema.fields[6].name, "rating")
        self.assertIsInstance(schema.fields[6].dataType, StructType)
        self.assertEqual(len(schema.fields[6].dataType.fields), 2)
        self.assertEqual(schema.fields[6].dataType.fields[0].name, "rate")
        self.assertEqual(schema.fields[6].dataType.fields[0].dataType, DecimalType(10, 2))

    def test_data_cleaning(self):
        # Criar DataFrame com dados simulados
        data = mock_api_response
        for item in data:
            item["price"] = Decimal(str(item["price"]))
            item["rating"]["rate"] = Decimal(str(item["rating"]["rate"]))
        
        schema = define_schema()
        df = self.spark.createDataFrame(data, schema)

        # Aplicar limpeza de dados
        df_clean = df.dropna(subset=["price", "category", "rating.rate"])
        df_clean = df_clean.withColumn("rate", col("rating.rate"))
        df_clean = df_clean.filter((col("price") >= 0) & (col("rate").between(0, 5)))
        
        # Filtros adicionais (preço >= 100 e rate >= 3.5)
        df_clean = df_clean.filter((col("price") >= 100) & (col("rate") >= 3.5))

        # Verificar resultados
        result = df_clean.collect()
        self.assertEqual(len(result), 2)  # Apenas 2 registros devem passar nos filtros
        self.assertEqual(result[0]["id"], 1)
        self.assertEqual(result[1]["id"], 3)

    def test_summary_aggregation(self):
        # Criar DataFrame com dados simulados
        data = [
            {"id": 1, "category": "electronics", "price": Decimal("150.00"), "rating": {"rate": Decimal("4.5"), "count": 100}},
            {"id": 2, "category": "electronics", "price": Decimal("120.00"), "rating": {"rate": Decimal("4.0"), "count": 50}},
            {"id": 3, "category": "clothing", "price": Decimal("200.00"), "rating": {"rate": Decimal("4.2"), "count": 75}}
        ]
        schema = define_schema()
        df = self.spark.createDataFrame(data, schema)

        # Aplicar sumarização
        df_summary = df.groupBy("category").agg(
            avg("price").alias("preco_medio"),
            avg("rating.rate").alias("avaliacao_media")
        ).withColumnRenamed("category", "categoria")

        result = df_summary.collect()
        result_dict = {row["categoria"]: (row["preco_medio"], row["avaliacao_media"]) for row in result}
        
        self.assertEqual(len(result), 2)  # Duas categorias
        self.assertAlmostEqual(result_dict["electronics"][0], 135.0, places=2)  # Média de preço
        self.assertAlmostEqual(result_dict["electronics"][1], 4.25, places=2)  # Média de avaliação
        self.assertAlmostEqual(result_dict["clothing"][0], 200.0, places=2)
        self.assertAlmostEqual(result_dict["clothing"][1], 4.2, places=2)

    

if __name__ == '__main__':
    unittest.main()