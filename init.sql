CREATE TABLE categoria_media (
    id SERIAL PRIMARY KEY,
    nome VARCHAR(100) NOT NULL,
    descricao TEXT
);

CREATE TABLE produtos (
    id SERIAL PRIMARY KEY,
    nome VARCHAR(100) NOT NULL,
    preco DECIMAL(10, 2) NOT NULL,
    categoria_id INTEGER REFERENCES categoria_media(id),
    criado_em TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);