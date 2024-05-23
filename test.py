import logging
import time
from time import sleep
import openai

def setup_custom_logger(log_file_name):
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    fileHandler = logging.FileHandler(log_file_name, mode='a')
    fileHandler.setFormatter(formatter)

    # 根据日志文件名，创建 Logger 实例；可以从不同的地方写入相同的 Log 文件
    logger = logging.getLogger(log_file_name)
    logger.setLevel(logging.INFO)
    logger.addHandler(fileHandler)
    logger.addHandler(logging.StreamHandler()) # Write to stdout as well
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    logger.info(f"Start logging: {time_}")

    return logger

logger = setup_custom_logger("output/test/log.txt")
from sparql_exe import execute_query, get_types, get_2hop_relations

def test_sparql():
    execute_query("PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?x\nWHERE {\nFILTER (?x != ns:m.0f8l9c)\nFILTER (!isLiteral(?x) OR lang(?x) = '' OR langMatches(lang(?x), 'en'))\nns:m.0f8l9c ns:location.location.adjoin_s ?y .\n?y ns:location.adjoining_relationship.adjoins ?x .\n?x ns:common.topic.notable_types ns:m.01mp .\n?x ns:location.statistical_region.size_of_armed_forces ?c .\n?c ns:measurement_unit.dated_integer.number \"101000\" . \n}", logger)
    get_types("m.0f8l9c", logger)
    get_2hop_relations("m.0f8l9c", logger)

def test_gpt():
    for idx in range(2):
        try:
            openai.api_key = "sk-ROoV4l0NELy39vGl45Fe5bDc7a954224A717BeFe4eCf10Ba"
            openai.api_base = "https://threefive.gpt7.link/v1"
            answer_modi = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=[{'role': 'user', 'content': "Introduce yourself in 10 words"}],
                temperature=0,
                max_tokens=256,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=["Question: "]
            )
            logger.info(f"answer_modi: {answer_modi}")
            break
        except Exception as e:
            logger.info(f"type_generator() exception: {e}; retrying idx: {idx}")
            sleep(3)

if __name__=="__main__":
    # test_sparql()
    test_gpt()