#!/usr/bin/env python3
"""Remove anúncios específicos e imóveis não prontos do dataset final.

Entrada esperada: dataset_final_auditavel.csv da etapa anterior.
Saídas:
- dataset_final_auditavel.csv (filtrado)
- dataset_final_modelagem.csv (14 colunas, derivado do auditável filtrado)
- removidos_imoveis_nao_prontos.csv (auditável + motivo)
- removidos_imoveis_nao_prontos_modelagem.csv (visão compacta)
- relatorio_remocao_imoveis_nao_prontos.txt

A lista foi construída a partir das evidências já preservadas no dataset auditável
(título e trechos da descrição usados no enriquecimento).
"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path

MODEL_COLUMNS = [
    "preco_venda",
    "tipo_imovel",
    "cidade",
    "bairro",
    "area_m2",
    "quartos",
    "banheiros",
    "vagas",
    "tem_suite",
    "tem_piscina",
    "tem_varanda",
    "tem_lazer",
    "tem_mobiliado",
    "tem_elevador",
]

# IDs explicitamente solicitados pelo usuário.
EXPLICIT_REMOVALS = {
    "1186410851": "anuncio_de_servico_de_construcao_pre_fabricada",
    "1515913123": "transferencia_de_financiamento_preco_nao_integral",
    "1516963456": "dois_imoveis_no_mesmo_anuncio",
}

# Imóveis com evidência explícita de obra, acabamento incompleto ou unidade ainda não pronta.
CONSTRUCTION_REMOVALS = {
    "1484399578": "apartamento_na_planta",
    "1512651899": "apartamento_na_planta",
    "1506258944": "imovel_em_fase_de_construcao",
    "1471489807": "unidade_a_reservar_com_imagens_ilustrativas",
    "1473136152": "prazo_de_obra_de_24_meses",
    "1511760737": "casa_semi_pronta",
    "1389741501": "casa_em_construcao_no_anuncio",
    "1370728753": "apartamento_na_planta",
    "1514570320": "conjunto_com_residencia_inacabada",
    "1439124807": "casa_em_construcao",
    "1390597095": "obra_iniciada_no_segundo_pavimento",
    "1326980036": "casa_em_construcao",
    "1498433562": "casa_em_fase_de_acabamento",
    "1432040804": "apartamento_com_obras_avancadas",
    "1516898003": "apartamento_em_fase_de_construcao",
    "1517067708": "apartamento_em_fase_de_construcao",
    "1496303239": "apartamento_faltando_acabamento",
    "1515031546": "apartamento_com_obras_iniciadas",
    "1485467334": "casa_nova_em_construcao",
    "1503323063": "casa_em_fase_de_acabamento",
    "1492147849": "parte_do_imovel_em_fase_de_acabamento",
    "1451076932": "casa_em_fase_de_acabamentos",
    "1451077842": "imovel_em_fase_de_acabamento",
    "1504974759": "quarto_em_fase_de_acabamento",
    "1469696436": "casa_em_construcao",
    "1438411437": "suite_inacabada",
    "1474982819": "projeto_com_imagens_ilustrativas_e_previsao",
    "1504197046": "pavimento_inacabado",
    "1516528048": "imovel_em_construcao",
    "1515160574": "empreendimento_com_status_em_construcao",
    "1515160699": "empreendimento_com_status_em_construcao",
    "1503878505": "dependencias_adicionais_em_construcao",
    "1512375048": "apartamentos_em_fase_de_obras",
    "1513370516": "casa_em_fase_final_de_acabamento",
    "1334801106": "casa_em_fase_final_de_acabamento",
    "1387374639": "casa_em_fase_final_de_construcao",
    "1479087999": "casa_em_fase_final_de_acabamento",
    "1503023345": "casa_em_fase_final_de_acabamento",
    "1459738417": "casa_em_fase_de_acabamento",
    "1516775031": "mansao_em_fase_final",
    "1511851996": "casa_em_ponto_de_acabamento",
}

# Entregas futuras, pré-lançamentos ou comercialização durante a obra.
FUTURE_DELIVERY_REMOVALS = {
    "1483145308": "pre_lancamento",
    "1515712544": "fase_de_pre_lancamento",
    "1342204735": "entrega_prevista_marco_2027",
    "1390109955": "entrega_prevista_julho_2026",
    "1514660272": "entrega_prevista_2028",
    "1513573093": "entrega_prevista_em_10_meses",
    "1516600126": "entrega_prevista_em_10_meses",
    "1514341476": "entrega_prevista_setembro_2026",
    "1517036691": "entrega_prevista_julho_2028",
    "1516236746": "entrega_prevista_julho_2026",
    "1487191381": "entrega_prevista_2028",
    "1510122119": "entrega_prevista_julho_2026",
    "1510122139": "entrega_prevista_julho_2026",
    "1511137654": "entrega_prevista_dezembro_2028",
    "1515446806": "entrega_prevista_2027",
    "1515601037": "entrega_prevista_novembro_2027",
    "1498138670": "entrega_prevista_dezembro_2026",
    "1511439342": "entrega_prevista_fevereiro_2028",
    "1515483299": "entrega_prevista_final_2026",
    "1511610623": "entrega_prevista_ainda_em_2026",
    "1472412764": "imovel_pronto_apenas_em_agosto_2026",
    "1513426106": "entrega_prevista_dezembro_2026",
    "1476784910": "entrega_indicada_para_2029",
    "1476786912": "entrega_indicada_para_2029",
    "1513426145": "entrega_prevista_dezembro_2028",
    "1513549368": "entrega_prevista_dezembro_2029",
    "1473276020": "anuncio_com_previsao_de_entrega_futura",
    "1448481988": "anuncio_com_previsao_de_entrega_sem_confirmacao_de_conclusao",
    "1321922610": "anuncio_com_previsao_de_entrega_sem_confirmacao_de_conclusao",
    "1324652561": "anuncio_com_previsao_de_entrega_sem_confirmacao_de_conclusao",
    "1472655620": "preco_e_condicao_durante_a_obra",
    "1501115285": "lancamento_com_estrutura_ainda_futura",
    "1501143732": "lancamento_com_estrutura_ainda_futura",
    "1239728696": "condominio_descrito_como_estrutura_futura",
}

REMOVAL_REASONS = {**EXPLICIT_REMOVALS, **CONSTRUCTION_REMOVALS, **FUTURE_DELIVERY_REMOVALS}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV sem cabeçalho: {path}")
        return list(reader.fieldnames), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def classify_group(reason: str) -> str:
    if reason in EXPLICIT_REMOVALS.values():
        return "remocao_especifica"
    if reason in CONSTRUCTION_REMOVALS.values():
        return "imovel_inacabado_ou_em_obra"
    return "entrega_futura_ou_pre_lancamento"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--copy-from-dir",
        type=Path,
        help="Diretório da versão anterior; copia os CSVs históricos de remoção, quando presentes.",
    )
    args = parser.parse_args()

    fieldnames, rows = read_csv(args.audit)
    required = {"id_anuncio", "url_anuncio", "titulo_raw", *MODEL_COLUMNS}
    missing = sorted(required - set(fieldnames))
    if missing:
        raise ValueError(f"Colunas obrigatórias ausentes: {missing}")

    ids_in_base = {row["id_anuncio"].strip() for row in rows}
    configured_missing = sorted(set(REMOVAL_REASONS) - ids_in_base)
    if configured_missing:
        raise ValueError(
            "IDs configurados para remoção não encontrados na base: " + ", ".join(configured_missing)
        )

    kept: list[dict[str, str]] = []
    removed: list[dict[str, str]] = []

    for row in rows:
        ad_id = row["id_anuncio"].strip()
        reason = REMOVAL_REASONS.get(ad_id)
        if reason is None:
            kept.append(row)
            continue

        enriched = dict(row)
        enriched["grupo_remocao_final"] = classify_group(reason)
        enriched["motivo_remocao_final"] = reason
        removed.append(enriched)

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    # Auditável final.
    write_csv(out / "dataset_final_auditavel.csv", fieldnames, kept)

    # Dataset final para modelagem sempre derivado do auditável filtrado.
    model_rows = [{column: row.get(column, "") for column in MODEL_COLUMNS} for row in kept]
    write_csv(out / "dataset_final_modelagem.csv", MODEL_COLUMNS, model_rows)

    removed_fields = fieldnames + ["grupo_remocao_final", "motivo_remocao_final"]
    write_csv(out / "removidos_imoveis_nao_prontos.csv", removed_fields, removed)

    compact_fields = [
        "id_anuncio",
        "url_anuncio",
        "titulo_raw",
        "grupo_remocao_final",
        "motivo_remocao_final",
        *MODEL_COLUMNS,
    ]
    write_csv(out / "removidos_imoveis_nao_prontos_modelagem.csv", compact_fields, removed)

    # Copia o histórico de remoções da versão anterior para manter a rastreabilidade completa.
    historical_files = [
        "removidos_duplicatas_exatas.csv",
        "removidos_motivos_revisao.csv",
        "removidos_revisao_final_manual.csv",
    ]
    if args.copy_from_dir:
        for filename in historical_files:
            source = args.copy_from_dir / filename
            if source.exists():
                shutil.copy2(source, out / filename)

    # Verificações de integridade.
    kept_ids = [row["id_anuncio"].strip() for row in kept]
    if len(kept_ids) != len(set(kept_ids)):
        raise RuntimeError("IDs duplicados permaneceram no dataset auditável final.")

    model_keys = [tuple(row.get(column, "") for column in MODEL_COLUMNS) for row in model_rows]
    if len(model_keys) != len(set(model_keys)):
        raise RuntimeError("Duplicatas exatas permaneceram nas 14 colunas do dataset de modelagem.")

    removed_ids = {row["id_anuncio"].strip() for row in removed}
    if removed_ids & set(kept_ids):
        raise RuntimeError("Há IDs simultaneamente nos arquivos mantido e removido.")

    counts = Counter(row["grupo_remocao_final"] for row in removed)
    report_lines = [
        "RELATORIO DE REMOCAO DE ANUNCIOS E IMOVEIS NAO PRONTOS",
        "=" * 78,
        f"Registros de entrada: {len(rows)}",
        f"Registros removidos nesta etapa: {len(removed)}",
        f"Registros finais para modelagem: {len(kept)}",
        "",
        "REMocoes POR GRUPO",
        "-" * 78,
    ]
    for group in sorted(counts):
        report_lines.append(f"{group}: {counts[group]}")

    report_lines.extend(
        [
            "",
            "VERIFICACOES",
            "-" * 78,
            f"IDs unicos no dataset final: {len(set(kept_ids))}",
            "Duplicatas exatas nas 14 colunas finais: 0",
            "IDs removidos ainda presentes no dataset final: 0",
            "Dataset de modelagem derivado diretamente do auditavel filtrado: sim",
            "",
            "SEMANTICA",
            "-" * 78,
            "A remocao de imoveis nao prontos foi baseada em evidencias explicitas",
            "preservadas no arquivo auditavel: imovel na planta, em construcao, em",
            "acabamento, inacabado, pre-lancamento ou com entrega futura.",
            "Anuncios que apenas usam a palavra 'lancamento', sem outra evidencia de",
            "obra ou entrega futura, nao foram removidos automaticamente.",
        ]
    )
    (out / "relatorio_remocao_imoveis_nao_prontos.txt").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    print("\n".join(report_lines))


if __name__ == "__main__":
    main()
