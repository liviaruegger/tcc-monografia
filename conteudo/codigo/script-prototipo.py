"""
***************************************************************************
*                                                                         *
*   This program is free software; you can redistribute it and/or modify  *
*   it under the terms of the GNU General Public License as published by  *
*   the Free Software Foundation; either version 2 of the License, or     *
*   (at your option) any later version.                                   *
*                                                                         *
***************************************************************************
"""

import sys
import os
import math
from typing import Any, Optional

import numpy as np

from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingContext,
    QgsProcessingException,
    QgsProcessingFeedback,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterFileDestination,
)

# GDAL/OGR imports - QGIS includes these libraries
from osgeo import gdal, osr


def get_utm_epsg_code(dataset):
    """
    Calcula o código EPSG da zona UTM correta para o centro do dataset.
    
    Args:
        dataset: Dataset GDAL aberto
        
    Returns:
        Tupla contendo (código EPSG UTM, longitude, latitude, zona UTM)
    """
    gt = dataset.GetGeoTransform()
    srs_origem = osr.SpatialReference(wkt=dataset.GetProjection())
    
    x_centro_pixel = dataset.RasterXSize / 2
    y_centro_pixel = dataset.RasterYSize / 2
    
    x_centro_mundo = gt[0] + (x_centro_pixel * gt[1]) + (y_centro_pixel * gt[2])
    y_centro_mundo = gt[3] + (x_centro_pixel * gt[4]) + (y_centro_pixel * gt[5])

    srs_wgs84 = osr.SpatialReference()
    srs_wgs84.ImportFromEPSG(4326)
    
    if int(gdal.__version__[0]) >= 3:
        srs_origem.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        srs_wgs84.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    transform = osr.CoordinateTransformation(srs_origem, srs_wgs84)
    point = transform.TransformPoint(x_centro_mundo, y_centro_mundo)
    lon = point[0]
    lat = point[1]

    zona_utm = math.floor((lon + 180) / 6) + 1
    
    if lat >= 0:
        epsg_code = 32600 + zona_utm
    else:
        epsg_code = 32700 + zona_utm
    
    return f"EPSG:{epsg_code}", lon, lat, zona_utm


def processar_geotiff_para_unity(input_path: str, output_raw_path: str, feedback: QgsProcessingFeedback) -> bool:
    """
    Executa o fluxo completo de conversão:
    1. Corta para quadrado (centro)
    2. Reprojeta para UTM (auto-detect)
    3. Converte para .raw 16-bit
    
    Args:
        input_path: Caminho do arquivo GeoTIFF de entrada
        output_raw_path: Caminho do arquivo .raw de saída
        feedback: Objeto de feedback do QGIS para logging
        
    Returns:
        True se a conversão foi bem-sucedida, False caso contrário
    """
    
    # Arquivos temporários em memória
    temp_cropped_path = f"/vsimem/{os.path.basename(input_path)}_cropped.tif"
    temp_warped_path = f"/vsimem/{os.path.basename(input_path)}_warped.tif"

    dataset = None
    cropped_ds = None
    warped_ds = None

    try:
        gdal.UseExceptions()
        
        # 0. Abrir o arquivo de entrada
        dataset = gdal.Open(input_path, gdal.GA_ReadOnly)
        if not dataset:
            feedback.pushConsoleInfo("Erro: Não foi possível abrir o arquivo de entrada.")
            return False

        feedback.pushConsoleInfo(f"--- Processando: {os.path.basename(input_path)} ---")

        # 1. CALCULAR O CORTE QUADRADO
        cols = dataset.RasterXSize
        rows = dataset.RasterYSize
        min_dim = min(cols, rows)
        
        x_offset = (cols - min_dim) // 2
        y_offset = (rows - min_dim) // 2
        
        feedback.pushConsoleInfo(f"Dimensões originais: {cols}x{rows}. Cortando para {min_dim}x{min_dim} a partir do centro.")

        gdal.Translate(temp_cropped_path, 
                         dataset, 
                         srcWin=[x_offset, y_offset, min_dim, min_dim],
                         format="GTiff")
        
        dataset = None 
        
        # 2. REPROJETAR PARA UTM
        cropped_ds = gdal.Open(temp_cropped_path, gdal.GA_ReadOnly)
        target_srs_utm, lon, lat, zona = get_utm_epsg_code(cropped_ds)
        
        feedback.pushConsoleInfo(f"Detectado Lon/Lat central: ({lon:.2f}, {lat:.2f})")
        feedback.pushConsoleInfo(f"Reprojetando para {target_srs_utm} (Zona {zona})...")
        
        warped_ds = gdal.Warp(temp_warped_path,
                               cropped_ds,
                               dstSRS=target_srs_utm,
                               resampleAlg=gdal.GRA_Bilinear,
                               format="GTiff")
        
        cropped_ds = None

        # 3. CONVERTER PARA UNITY .RAW (16-BIT)
        feedback.pushConsoleInfo("Iniciando conversão para .raw 16-bit...")

        band = warped_ds.GetRasterBand(1)
        final_cols = warped_ds.RasterXSize
        final_rows = warped_ds.RasterYSize
        
        feedback.pushConsoleInfo(f"Resolução final (para o Unity): {final_cols}x{final_rows}")
        
        data = band.ReadAsArray().astype(np.float32)
        
        nodata_value = band.GetNoDataValue()
        if nodata_value is not None:
            mask = data != nodata_value
            if not mask.any():
                 feedback.pushConsoleInfo("Erro: O arquivo parece conter apenas 'NoData'.")
                 return False
            min_height = np.min(data[mask])
            data[~mask] = min_height
        else:
            min_height = np.min(data)
            
        max_height = np.max(data)

        feedback.pushConsoleInfo(f"Altura Mínima Real (UTM): {min_height:.2f}m")
        feedback.pushConsoleInfo(f"Altura Máxima Real (UTM): {max_height:.2f}m")
        feedback.pushConsoleInfo(f"Variação (Terrain Height no Unity): {max_height - min_height:.2f}m")
        
        if max_height == min_height:
            data_uint16 = np.zeros_like(data, dtype=np.uint16)
        else:
            data_normalized = (data - min_height) / (max_height - min_height)
            data_uint16 = (data_normalized * 65535).astype(np.uint16)
            
        if sys.byteorder == 'big':
            data_uint16.byteswap(inplace=True)

        # 4. SALVAR O ARQUIVO .RAW FINAL
        with open(output_raw_path, 'wb') as f:
            data_uint16.tofile(f)

        feedback.pushConsoleInfo(f"\nSUCESSO! Arquivo salvo em: {output_raw_path}")
        return True

    except Exception as e:
        feedback.pushConsoleInfo(f"Ocorreu um erro durante o processamento: {e}")
        return False
    
    finally:
        # 5. LIMPEZA
        dataset = None
        cropped_ds = None
        warped_ds = None
        try:
            gdal.Unlink(temp_cropped_path)
        except: 
            pass
        try:
            gdal.Unlink(temp_warped_path)
        except: 
            pass


class ConvertToUnityRaw(QgsProcessingAlgorithm):
    """
    Algoritmo de processamento do QGIS que converte um GeoTIFF
    em um arquivo .raw 16-bit para o Unity, aplicando corte quadrado
    e reprojeção UTM automática.
    
    Este script foi desenvolvido a partir do template oficial do QGIS
    para algoritmos de processamento, adaptando a estrutura documentada
    do template para incorporar a lógica de conversão de dados geoespaciais.
    """

    # Constants used to refer to parameters and outputs. They will be
    # used when calling the algorithm from another algorithm, or when
    # calling from the QGIS console.
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"

    def name(self) -> str:
        """
        Returns the algorithm name, used for identifying the algorithm. This
        string should be fixed for the algorithm, and must not be localised.
        The name should be unique within each provider. Names should contain
        lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return "convertunityraw"

    def displayName(self) -> str:
        """
        Returns the translated algorithm name, which should be used for any
        user-visible display of the algorithm name.
        """
        return self.tr("Converter para Unity RAW (UTM, Quadrado)")

    def group(self) -> str:
        """
        Returns the name of the group this algorithm belongs to. This string
        should be localised.
        """
        return self.tr("Unity Converter")

    def groupId(self) -> str:
        """
        Returns the unique ID of the group this algorithm belongs to. This
        string should be fixed for the algorithm, and must not be localised.
        The group id should be unique within each provider. Group id should
        contain lowercase alphanumeric characters only and no spaces or other
        formatting characters.
        """
        return "unityconverter"

    def shortHelpString(self) -> str:
        """
        Returns a localised short helper string for the algorithm. This string
        should provide a basic description about what the algorithm does and the
        parameters and outputs associated with it.
        """
        return self.tr(
            "Converte um arquivo GeoTIFF de elevação para formato RAW 16-bit "
            "compatível com Unity. Aplica corte quadrado (centro) e reprojeção "
            "automática para UTM."
        )

    def initAlgorithm(self, config: Optional[dict[str, Any]] = None):
        """
        Here we define the inputs and output of the algorithm, along
        with some other properties.
        """
        
        # Input raster layer (GeoTIFF)
        self.addParameter(
            QgsProcessingParameterRasterLayer(
                self.INPUT,
                self.tr("Camada de Heightmap de Entrada (GeoTIFF)")
            )
        )

        # Output file destination (.raw)
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT,
                self.tr("Arquivo de Saída .raw"),
                self.tr("Unity RAW Files (*.raw)")
            )
        )

    def processAlgorithm(
        self,
        parameters: dict[str, Any],
        context: QgsProcessingContext,
        feedback: QgsProcessingFeedback,
    ) -> dict[str, Any]:
        """
        Here is where the processing itself takes place.
        """
        
        # Retrieve the input raster layer
        input_layer = self.parameterAsRasterLayer(parameters, self.INPUT, context)
        
        # If source was not found, throw an exception to indicate that the algorithm
        # encountered a fatal error
        if input_layer is None:
            raise QgsProcessingException(
                self.invalidSourceError(parameters, self.INPUT)
            )
        
        input_path = input_layer.source()
        
        # Get the output file path
        output_path = self.parameterAsFileOutput(parameters, self.OUTPUT, context)
        
        # Call the main processing function
        success = processar_geotiff_para_unity(input_path, output_path, feedback)
        
        if not success:
            raise QgsProcessingException(
                "Falha durante o processamento. Verifique os logs para mais detalhes."
            )
        
        # Return the results of the algorithm
        return {self.OUTPUT: output_path}

    def createInstance(self):
        """
        Necessary for QGIS to create an instance of the class.
        """
        return ConvertToUnityRaw()

    def tr(self, string: str) -> str:
        """
        Translation function (can be extended for i18n support).
        """
        return string

