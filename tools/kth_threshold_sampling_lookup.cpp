#include <iostream>
#include <optional>
#include <unordered_set>
#include <queue>
#include <random>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>
#include <sparsehash/dense_hash_set>
#include <sparsehash/dense_hash_map>
#include <bits/stdc++.h>

#include "boost/algorithm/string/classification.hpp"
#include "boost/algorithm/string/split.hpp"
#include <boost/functional/hash.hpp>
#include <boost/interprocess/file_mapping.hpp>
#include <boost/interprocess/mapped_region.hpp>

#include "mio/mmap.hpp"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

#include "mappable/mapper.hpp"

#include "app.hpp"
#include "cursor/max_scored_cursor.hpp"
#include "index_types.hpp"
#include "io.hpp"
#include "query/algorithm.hpp"
#include "util/util.hpp"
#include "wand_data_compressed.hpp"
#include "wand_data_raw.hpp"

#include "query/algorithm.hpp"
#include "query/queries.hpp"
#include "scorer/scorer.hpp"

#include "CLI/CLI.hpp"

#define MaxQueryLen 16

using namespace pisa;
namespace bip = boost::interprocess;
using term_id_type = uint32_t;
//using ext::hash;
using google::dense_hash_set;
using google::dense_hash_map;

//string single_freq_path = "/home/jg6226/data/Hit_Ratio_Project/Lexicon/CW09B.fwd.terms";
string single_gram_path = "/home/jg6226/data/Hit_Ratio_Project/Lexicon/CW09B.fwd.terms";
string single_prefix_path = "/ssd3/jg6226/data/Hit_Ratio_Project/Real_Time_Query_System/First_Layer_Index/single_with_termscore/single_prefix";
string single_lexicon_path = "/ssd3/jg6226/data/Hit_Ratio_Project/Real_Time_Query_System/First_Layer_Index/single_with_termscore/single_lexicon.txt";

//string duplet_freq_path = "/ssd2/home/bmmliu/logBaseFreq/2_term_freq_1.txt";
string duplet_gram_path = "/home/jg6226/data/Hit_Ratio_Project/Real_Time_Query_System/Prefix_Grams/duplet_cleaned.txt";
string duplet_prefix_path = "/ssd3/jg6226/data/Hit_Ratio_Project/Real_Time_Query_System/First_Layer_Index/duplet_with_termscore/duplet_prefix";
string duplet_lexicon_path = "/ssd3/jg6226/data/Hit_Ratio_Project/Real_Time_Query_System/First_Layer_Index/duplet_with_termscore/duplet_lexicon.txt";

//string triplet_freq_path = "/ssd2/home/bmmliu/logBaseFreq/3_term_freq_1.txt";
string triplet_gram_path = "/home/jg6226/data/Hit_Ratio_Project/Real_Time_Query_System/Prefix_Grams/triplet_cleaned.txt";
string triplet_prefix_path = "/ssd3/jg6226/data/Hit_Ratio_Project/Real_Time_Query_System/First_Layer_Index/triplet_with_termscore/triplet_prefix";
string triplet_lexicon_path = "/ssd3/jg6226/data/Hit_Ratio_Project/Real_Time_Query_System/First_Layer_Index/triplet_with_termscore/triplet_lexicon.txt";

//string quadruplet_freq_path = "/ssd2/home/bmmliu/logBaseFreq/4_term_freq_1.txt";
string quadruplet_gram_path = "/home/jg6226/data/Hit_Ratio_Project/Real_Time_Query_System/Prefix_Grams/quadruplet_cleaned.txt";
string quadruplet_prefix_path = "/ssd3/jg6226/data/Hit_Ratio_Project/Real_Time_Query_System/First_Layer_Index/quadruplet_with_termscore/quadruplet_prefix";
string quadruplet_lexicon_path = "/ssd3/jg6226/data/Hit_Ratio_Project/Real_Time_Query_System/First_Layer_Index/quadruplet_with_termscore/quadruplet_lexicon.txt";

ifstream single_prefix_binary;

ifstream duplet_prefix_binary;

ifstream triplet_prefix_binary;

ifstream quadruplet_prefix_binary;

class InvertedIndex {
public:
  InvertedIndex(const std::string& input_index_path, const std::string& input_index_name, const std::string& term_lexicon_path) {
      index_path = input_index_path;
      index_name = input_index_name;

      // Load docs and scores
      //load_vector_from_file(docs, index_path + '/' + index_name + ".docs");
      /*string docsFilePath = index_path + '/' + index_name + ".docs";
      bip::file_mapping docsFileMapping(docsFilePath.c_str(), bip::read_only);
      bip::mapped_region docsMappedRegion(docsFileMapping, bip::read_only);
      docs = static_cast<uint32_t*>(docsMappedRegion.get_address());
      docs_length = docsMappedRegion.get_size() / sizeof(uint32_t);

      clog << docs_length << endl;
      clog << docs[0] << endl;
      clog << "load docs finished" << endl;

      string scoresFilePath = index_path + '/' + index_name + ".scores";
      bip::file_mapping scoresFileMapping(scoresFilePath.c_str(), bip::read_only);
      bip::mapped_region scoresMappedRegion(scoresFileMapping, bip::read_only);
      scores = static_cast<uint32_t*>(scoresMappedRegion.get_address());
      scores_length = scoresMappedRegion.get_size() / sizeof(uint32_t);

      clog << scores_length << endl;
      clog << scores[0] << endl;
      clog << "load scores finished" << endl;*/

      // Load term lexicon
      uint64_t cnt_line = 0;
      std::ifstream f_term_lex(term_lexicon_path);
      std::string line;
      while (std::getline(f_term_lex, line)) {
          std::string term = line;
          dict_term_termid[term] = cnt_line;
          dict_termid_term[cnt_line] = term;
          cnt_line++;
      }
      f_term_lex.close();

      std::cout << "Total number terms: " << cnt_line << std::endl;
  }

  vector<string> split (const string &s, char delim) {
      vector<string> result;
      stringstream ss (s);
      string item;

      while (getline (ss, item, delim)) {
          result.push_back (item);
      }

      return result;
  }

  string strip(string &inpt) {
      auto start_it = inpt.begin();
      auto end_it = inpt.rbegin();
      while (std::isspace(*start_it))
          ++start_it;
      while (std::isspace(*end_it))
          ++end_it;
      return std::string(start_it, end_it.base());
  }

  void load_vector_from_file(std::vector<uint32_t>& vec, const std::string& file_path) {
      std::ifstream input_file(file_path, std::ios::binary);
      if (!input_file) {
          std::cerr << "Error opening file: " << file_path << std::endl;
          exit(1);
      }

      input_file.seekg(0, std::ios::end);
      std::streampos file_size = input_file.tellg();
      input_file.seekg(0, std::ios::beg);

      vec.resize(file_size / sizeof(uint32_t));
      input_file.read(reinterpret_cast<char*>(vec.data()), file_size);
      input_file.close();
  }

  void initialize(const std::string& index_lexicon_path, const std::string& blocklast_file_path) {
      load_index_lexicon(index_lexicon_path);
      clog << "load index_lexicon finished" << endl;
      load_blocklast_metadata(blocklast_file_path);
      clog << "load blocklast_metadata finished" << endl;
  }

  void load_index_lexicon(const std::string& index_lexicon_path) {
      std::ifstream f_index_lex(index_lexicon_path);
      std::string line;
      while (std::getline(f_index_lex, line)) {
          vector<string> line_list = split(strip(line), ' ');
          index_lex[stoi(line_list[0])] = {stoul(line_list[1]), stoul(line_list[2])};
      }
      f_index_lex.close();
  }

  vector<uint64_t> lookup_bm25_score_skip_block_fast(string &term, std::vector<uint64_t> &lst_did, int block_size = 1024) {


      string docsFilePath = index_path + '/' + index_name + ".docs";
      bip::file_mapping docsFileMapping(docsFilePath.c_str(), bip::read_only);
      bip::mapped_region docsMappedRegion(docsFileMapping, bip::read_only);
      uint32_t* docs = static_cast<uint32_t*>(docsMappedRegion.get_address());

      string scoresFilePath = index_path + '/' + index_name + ".scores";
      bip::file_mapping scoresFileMapping(scoresFilePath.c_str(), bip::read_only);
      bip::mapped_region scoresMappedRegion(scoresFileMapping, bip::read_only);
      uint32_t* scores = static_cast<uint32_t*>(scoresMappedRegion.get_address());

      string lastdidFilePath = index_path + '/' + index_name + ".blocklast";
      bip::file_mapping lastdidFileMapping(lastdidFilePath.c_str(), bip::read_only);
      bip::mapped_region lastdidMappedRegion(lastdidFileMapping, bip::read_only);
      uint32_t* lastdid = static_cast<uint32_t*>(lastdidMappedRegion.get_address());

      uint64_t termid = dict_term_termid[term];
      auto offset_tuple = index_lex[termid];
      uint64_t start_i = offset_tuple.first;
      uint64_t end_i = offset_tuple.second;

      std::vector<uint64_t> lst_score_result;

      uint64_t current_i = start_i + 1;
      uint64_t block_idx = 0;
      auto block_info_list = blocklast_lex[termid];

      uint64_t block_num = lastdid[block_info_list.first];
      uint64_t blocklast_real_start_index = block_info_list.first + 1;

      /*std::cout << "termid: " << termid << std::endl;
      std::cout << "start_i: " << start_i << std::endl;
      std::cout << "end_i: " << end_i << std::endl;
      std::cout << "current_i: " << current_i << std::endl;
      std::cout << "block_idx: " << block_idx << std::endl;
      std::cout << "block_info_list: (" << block_info_list.first << ", " << block_info_list.second << ")" << std::endl;
      std::cout << "block_num: " << block_num << std::endl;
      std::cout << "blocklast_real_start_index: " << blocklast_real_start_index << std::endl;*/

      uint64_t did_index = 0;
      uint64_t block_offset = 0;
      while (did_index < lst_did.size()) {
          uint64_t did = lst_did[did_index];
          while (block_idx < block_num - 1) {
              if (lastdid[blocklast_real_start_index + block_idx] < did) {
                  block_idx += 1;
                  block_offset = 0;
              } else if (lastdid[blocklast_real_start_index + block_idx] >= did) {
                  break;
              }
          }

          bool complete_tag = false;
          for (uint64_t posting_idx = block_offset + 2 + current_i + block_size * block_idx;
               posting_idx < std::min(end_i + 2, 2 + current_i + block_size * (block_idx + 1));
               posting_idx++) {

              if (docs[posting_idx] == did) {
                  uint64_t score = scores[posting_idx - 2];
                  lst_score_result.push_back(score);
                  did_index += 1;
                  complete_tag = true;
                  block_offset = posting_idx - (2 + current_i + block_size * block_idx);
                  break;
              } else if (docs[posting_idx] > did) {
                  uint64_t score = 0;
                  lst_score_result.push_back(score);
                  did_index += 1;
                  complete_tag = true;
                  block_offset = posting_idx - (2 + current_i + block_size * block_idx);
                  break;
              }
          }

          if (!complete_tag) {
              uint64_t score = 0;
              lst_score_result.push_back(score);
              did_index += 1;
          }
      }

      return lst_score_result;
  }

  void load_blocklast_metadata(const std::string& blocklast_file_path) {
      //load_vector_from_file(lastdid, blocklast_file_path);

      bip::file_mapping lastdidFileMapping(blocklast_file_path.c_str(), bip::read_only);
      bip::mapped_region lastdidMappedRegion(lastdidFileMapping, bip::read_only);
      uint32_t* lastdid = static_cast<uint32_t*>(lastdidMappedRegion.get_address());
      long lastdid_length = lastdidMappedRegion.get_size() / sizeof(uint32_t);

      clog << lastdid_length << endl;
      clog << lastdid[0] << endl;

      uint32_t termid = 0;
      uint64_t i = 0;
      while (i < lastdid_length) {
          uint64_t current_block_num = lastdid[i];
          uint64_t start_offset = i;
          uint64_t end_offset = i + current_block_num;
          blocklast_lex[termid] = {start_offset, end_offset};

          termid += 1;
          i += current_block_num;
      }
  }

private:
    std::string index_path;
    std::string index_name;
    //std::vector<uint32_t> docs;
    //std::vector<uint32_t> scores;
    //std::vector<uint32_t> lastdid;

    std::unordered_map<std::string, uint64_t> dict_term_termid;
    std::unordered_map<uint64_t, std::string> dict_termid_term;
    std::unordered_map<uint64_t, std::pair<uint64_t, uint64_t>> index_lex;
    std::unordered_map<uint64_t, std::pair<uint64_t, uint64_t>> blocklast_lex;
};

struct Posting
{
    uint64_t did;
    short total_score;
    vector<short> scores;
    string term_string;
    int rank;

    Posting(uint64_t did, short total_score, vector<short> scores, string term_string, int rank)
        : did(did), total_score(total_score), scores(scores), term_string(term_string), rank(rank) {}
};

struct ComparePostingScore {
    bool operator()(Posting const& p1, Posting const& p2)
    {
        // return "true" if "p1" is ordered
        // before "p2", for example:
        return p1.total_score < p2.total_score;
    }
};

struct eqstr {
    bool operator()(const string &s1, const string &s2) const
    {
        return s1.compare(s2) == 0;
    }
};

struct eq64int
{
    bool operator()(const uint64_t &i1, const uint64_t &i2) const
    {
        return i1 == i2;
    }
};

struct eq32int
{
    bool operator()(const uint32_t &i1, const uint32_t &i2) const
    {
        return i1 == i2;
    }
};


double nCr(double n, double r) {
    double sum = 1;

    for(int i = 1; i <= r; i++){
        sum = sum * (n - r + i) / i;
    }

    //return (int)sum;
    return std::exp(std::lgamma(n + 1)- std::lgamma(r + 1) - std::lgamma(n - r + 1));
}

double calculateO(int k, int kPrime, double s) {
    double res = 0.0;

    for (int i = kPrime; i < k; i++) {
        res += nCr(k - 1, i) * std::pow(s, i) * std::pow(1 - s, k - i - 1);
    }

    return res;
}

int getKPrime(int k, float s, float target_O) {

    for (int k_prime = 1; k_prime < k; k_prime++) {
        if (calculateO(k, k_prime, s) <= target_O) {
            return k_prime;
        }
    }

    return -1;
}

unordered_set<uint64_t> getSampleDid(string sample_file_path) {
    /*int sampleSize = didMax * samplePercent;
    std::vector<int> numbers(didMax+1);
    std::iota(numbers.begin(), numbers.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(numbers.begin(), numbers.end(), g);
    std::unordered_set<uint64_t> result(numbers.begin(), numbers.begin() + sampleSize);*/

    std::unordered_set<uint64_t> result;
    string did_string;
    ifstream MyReadFile(sample_file_path);
    while (getline (MyReadFile, did_string)) {
        result.insert(stoull(did_string));
    }

    // Close the file
    MyReadFile.close();
    return result;
}

std::set<uint32_t> parse_tuple(std::string const& line, size_t k)
{
    std::vector<std::string> term_ids;
    boost::algorithm::split(term_ids, line, boost::is_any_of(" \t"));
    if (term_ids.size() != k) {
        throw std::runtime_error(fmt::format(
            "Wrong number of terms in line: {} (expected {} but found {})", line, k, term_ids.size()));
    }

    std::set<uint32_t> term_ids_int;
    for (auto&& term_id: term_ids) {
        try {
            term_ids_int.insert(std::stoi(term_id));
        } catch (...) {
            throw std::runtime_error(
                fmt::format("Cannot convert {} to int in line: {}", term_id, line));
        }
    }
    return term_ids_int;
}

vector<string> split (const string &s, char delim) {
    vector<string> result;
    stringstream ss (s);
    string item;

    while (getline (ss, item, delim)) {
        result.push_back (item);
    }

    return result;
}

vector<Query> make_exist_term_queries(std::string exist_term_file) {
    std::vector<::pisa::Query> q;
    std::string m_term_lexicon = "/home/bmmliu/data/cw09b/CW09B.fwd.termlex";
    auto parse_query = resolve_query_parser(q, m_term_lexicon, std::nullopt, std::nullopt);
    std::ifstream is(exist_term_file);
    io::for_each_line(is, parse_query);
    return q;
}

bool cmpDidTPair(pair<uint64_t, short>& a, pair<uint64_t, short>& b){
    return a.second > b.second;
}

bool ifDupTerm(vector<uint32_t> terms) {
    unordered_set<uint32_t> memo;
    for (uint32_t term : terms) {
        if (memo.find(term) != memo.end()) {
            return true;
        }
        memo.insert(term);
    }

    return false;
}


vector<string> getAllPossibleComb(vector<uint32_t> &terms, int termConsidered) {
    vector<string> retVal_string;

    if (terms.size() >= 1 && termConsidered >= 1) {
        for (int i = 0; i < terms.size(); ++i) {
            retVal_string.push_back(to_string(terms[i]));
        }
    }
    if (terms.size() >= 2 && termConsidered >= 2) {
        for (int i = 0; i < terms.size(); ++i) {
            for (int j = i + 1; j < terms.size(); ++j) {
                retVal_string.push_back(to_string(terms[i]) + "-" + to_string(terms[j]));
            }
        }
    }

    if (terms.size() >= 3 && termConsidered >= 3) {
        for (int i = 0; i < terms.size(); ++i) {
            for (int j = i + 1; j < terms.size(); ++j) {
                for (int s = j + 1; s < terms.size(); ++s) {
                    retVal_string.push_back(to_string(terms[i]) + "-" + to_string(terms[j]) + "-" + to_string(terms[s]));
                }
            }
        }
    }

    if (terms.size() >= 4 && termConsidered >= 4) {
        for (int i = 0; i < terms.size(); ++i) {
            for (int j = i + 1; j < terms.size(); ++j) {
                for (int s = j + 1; s < terms.size(); ++s) {
                    for (int t = s + 1; t < terms.size(); ++t) {
                        retVal_string.push_back(to_string(terms[i]) + "-" + to_string(terms[j]) + "-" + to_string(terms[s]) + "-" + to_string(terms[t]));
                    }
                }
            }
        }
    }

    return retVal_string;
}

vector<uint32_t> getTermsFromString (string &termStr) {
    vector<string> termsVecStr = split(termStr, '-');
    vector<uint32_t> terms;
    for (string str : termsVecStr) {
        terms.push_back(static_cast<uint32_t>(std::stoul(str)));
    }

    return terms;
}

string term_to_string (vector<uint32_t> &terms) {
    string terms_string = "";
    for (uint32_t term_id: terms) {
        terms_string += to_string(term_id) + "-";
    }
    terms_string.pop_back();

    return terms_string;
}

void load_lexicon(dense_hash_map<string, pair<int64_t, int64_t>, hash<string>, eqstr> &lex_map, string &freq_path, string &gram_path, string &lexicon_path, int gram_size) {
    vector<Query> exist_terms_queries = make_exist_term_queries(gram_path);
    vector<Query> freq_terms_queries = make_exist_term_queries(freq_path);
    unordered_set<string> freq_terms_string_set;

    for (auto const& query: freq_terms_queries) {
        vector<uint32_t> freq_query_terms = query.terms;
        string freq_term_id_string = term_to_string(freq_query_terms);
        freq_terms_string_set.insert(freq_term_id_string);
    }

    ifstream in_lex(lexicon_path);

    int count = 0;

    for (auto const& query: exist_terms_queries) {

        vector<uint32_t> unsorted_terms = query.terms;
        string term_id_string = term_to_string(unsorted_terms);

        if (gram_size != unsorted_terms.size() || freq_terms_string_set.find(term_id_string) == freq_terms_string_set.end()) {
            //clog << "loading non stemmed query" << endl;
            string lexline;
            std::getline(in_lex, lexline);
            continue;
        }

        string lexline;
        vector<string> lex_pair;

        std::getline(in_lex, lexline);
        boost::split(lex_pair, lexline, boost::is_any_of(" "), boost::token_compress_on);
        lex_map[term_id_string] = pair<int64_t, int64_t>(std::stoll(lex_pair[0], nullptr, 0), std::stoll(lex_pair[1], nullptr, 0));

        count++;
        if (count % 10000000 == 0) {
            clog << count << " " << gram_size << " term combine loaded" << endl;
        }
    }

    in_lex.close();
}

// return 1 if push successfully
int insert_posting_to_heap(priority_queue<Posting, vector<Posting>, ComparePostingScore> &posting_max_heap, string &term_string, int rank, dense_hash_map<string, pair<int64_t, int64_t>, hash<string>, eqstr> &lex_map, unordered_set<uint64_t> &SampleDid) {
    vector<uint32_t> terms = getTermsFromString(term_string);
    int64_t start_pos = lex_map[term_string].first;
    int64_t end_pos = lex_map[term_string].second;
    int64_t cur_pos = start_pos + 4;
    cur_pos = cur_pos + rank * (4 + terms.size());

    while (cur_pos < end_pos) {
        int did;
        vector<short> scores;
        short total_score = 0;
        if (terms.size() == 1) {
            single_prefix_binary.seekg(cur_pos);
            single_prefix_binary.read(reinterpret_cast<char*>(&did), 4);
            for (int i = 0; i < terms.size(); i++) {
                unsigned char score;
                single_prefix_binary.read(reinterpret_cast<char*>(&score), 1);
                scores.push_back((short)score);
                total_score += (short)score;
            }
        } else if (terms.size() == 2) {
            duplet_prefix_binary.seekg(cur_pos);
            duplet_prefix_binary.read(reinterpret_cast<char*>(&did), 4);
            for (int i = 0; i < terms.size(); i++) {
                unsigned char score;
                duplet_prefix_binary.read(reinterpret_cast<char*>(&score), 1);
                scores.push_back((short)score);
                total_score += (short)score;
            }
        } else if (terms.size() == 3) {
            triplet_prefix_binary.seekg(cur_pos);
            triplet_prefix_binary.read(reinterpret_cast<char*>(&did), 4);
            for (int i = 0; i < terms.size(); i++) {
                unsigned char score;
                triplet_prefix_binary.read(reinterpret_cast<char*>(&score), 1);
                scores.push_back((short)score);
                total_score += (short)score;
            }
        } else if (terms.size() == 4) {
            quadruplet_prefix_binary.seekg(cur_pos);
            quadruplet_prefix_binary.read(reinterpret_cast<char*>(&did), 4);
            for (int i = 0; i < terms.size(); i++) {
                unsigned char score;
                quadruplet_prefix_binary.read(reinterpret_cast<char*>(&score), 1);
                scores.push_back((short)score);
                total_score += (short)score;
            }
        } else {
            cerr << "Wrong posting was considered" << endl;
        }

        if (SampleDid.find(did) == SampleDid.end()) {
            rank++;
            scores.clear();
            total_score = 0;
            cur_pos = cur_pos + 4 + terms.size();
            continue;
        }
        posting_max_heap.push(Posting(did, total_score, scores, term_string, rank));
        return 1;
    }
    return 0;
}

template <typename IndexType, typename WandType>
void kt_thresholds(
    const std::string& index_filename,
    const std::string& wand_data_filename,
    const std::vector<Query>& queries,
    std::string const& type,
    ScorerParams const& scorer_params,
    uint64_t k,
    bool quantized,
    std::optional<std::string> pairs_filename,
    std::optional<std::string> triples_filename,
    std::optional<std::string> exist_term_filename,
    bool all_pairs,
    bool all_triples)
{
    string index_path = "/home/jg6226/data/Hit_Ratio_Project/Uncompressed_Index_Quantized";
    string index_name = "CW09B.quantized.url.inv";
    string term_lexicon_path = "/home/jg6226/data/Hit_Ratio_Project/Lexicon/CW09B.fwd.terms";
    string index_lexicon_path = index_path + "/CW09B.quantized.url.inv.indexlex";
    string blocklast_file_path = index_path + "/CW09B.quantized.url.inv.blocklast";

    InvertedIndex invertedIndex(index_path, index_name, term_lexicon_path);
    invertedIndex.initialize(index_lexicon_path, blocklast_file_path);

    string term_to_lookup = "oklahoma";
    vector<uint64_t> didList = {0, 1, 17342170, 43409139, 43409140};
    clog << "Look up start" << endl;
    vector<uint64_t> lookup_result = invertedIndex.lookup_bm25_score_skip_block_fast(term_to_lookup, didList);
    for (auto score : lookup_result) {
        cout << score << ", ";
    }
    cout << endl;

    term_to_lookup = "certif";
    lookup_result = invertedIndex.lookup_bm25_score_skip_block_fast(term_to_lookup, didList);
    for (auto score : lookup_result) {
        cout << score << ", ";
    }
    cout << endl;

    term_to_lookup = "teach";
    lookup_result = invertedIndex.lookup_bm25_score_skip_block_fast(term_to_lookup, didList);
    for (auto score : lookup_result) {
        cout << score << ", ";
    }
    cout << endl;

    return;

    IndexType index;
    mio::mmap_source m(index_filename.c_str());
    mapper::map(index, m);

    WandType wdata;

    auto scorer = scorer::from_params(scorer_params, wdata);

    mio::mmap_source md;
    std::error_code error;
    md.map(wand_data_filename, error);
    if (error) {
        spdlog::error("error mapping file: {}, exiting...", error.message());
        std::abort();
    }
    mapper::map(wdata, md, mapper::map_flags::warmup);

    using Pair = std::set<uint32_t>;
    std::unordered_set<Pair, boost::hash<Pair>> pairs_set;

    using Triple = std::set<uint32_t>;
    std::unordered_set<Triple, boost::hash<Triple>> triples_set;

    std::string line;
    if (all_pairs) {
        spdlog::info("All pairs are available.");
    }
    if (pairs_filename) {
        std::ifstream pin(*pairs_filename);
        while (std::getline(pin, line)) {
            pairs_set.insert(parse_tuple(line, 2));
        }
        spdlog::info("Number of pairs loaded: {}", pairs_set.size());
    }

    if (all_triples) {
        spdlog::info("All triples are available.");
    }
    if (triples_filename) {
        std::ifstream trin(*triples_filename);
        while (std::getline(trin, line)) {
            triples_set.insert(parse_tuple(line, 3));
        }
        spdlog::info("Number of triples loaded: {}", triples_set.size());
    }

    // get all kinds of combinations
    //int numberOfTerms = 3;
    //int d = k * 10;
    int getDandTFromFileNameFlag = 1;
    vector<string> argStr;

    if (exist_term_filename && getDandTFromFileNameFlag) {
        argStr = split(*exist_term_filename, ',');
    }

    int sample_percentage = atoi(argStr[0].c_str());

    string sample_did_file_name = "/ssd2/home/bmmliu/SampleDid/" + to_string(sample_percentage) + "PercentSampleDid.txt";
    int budget = atoi(argStr[1].c_str());
    int termConsidered = 4;
    float target_over_estimate_rate = atof(argStr[2].c_str());

    string freq = argStr[3].c_str();
    string sin_freq_file = "/home/jg6226/data/Hit_Ratio_Project/Lexicon/CW09B.fwd.terms";
    string dup_freq_file = "/ssd2/home/bmmliu/logBaseFreq/2_term_freq_" + string(1, freq[1]) + ".txt";
    string tri_freq_file = "/ssd2/home/bmmliu/logBaseFreq/3_term_freq_" + string(1, freq[2]) + ".txt";
    string qud_freq_file = "/ssd2/home/bmmliu/logBaseFreq/4_term_freq_" + string(1, freq[3]) + ".txt";

    unordered_set<uint64_t> SampleDid = getSampleDid(sample_did_file_name);

    int KPrime = getKPrime(k, sample_percentage / 100.0, target_over_estimate_rate);
    cout << "k = " << k << endl;
    cout << "budget = " << budget << endl;
    cout << "target O = " << target_over_estimate_rate << endl;
    cout << "k prime = " << KPrime << endl;

    dense_hash_map<string, pair<int64_t, int64_t>, hash<string>, eqstr> lex_map;
    lex_map.set_empty_key("NULL");

    load_lexicon(lex_map, sin_freq_file, single_gram_path, single_lexicon_path, 1);
    if (termConsidered >= 2) {
        load_lexicon(lex_map, dup_freq_file, duplet_gram_path, duplet_lexicon_path, 2);
    }
    if (termConsidered >= 3) {
        load_lexicon(lex_map, tri_freq_file, triplet_gram_path, triplet_lexicon_path, 3);
    }
    if (termConsidered >= 4) {
        load_lexicon(lex_map, qud_freq_file, quadruplet_gram_path, quadruplet_lexicon_path, 4);
    }

    single_prefix_binary.open(single_prefix_path, std::ios::in | std::ios::binary);
    duplet_prefix_binary.open(duplet_prefix_path, std::ios::in | std::ios::binary);
    triplet_prefix_binary.open(triplet_prefix_path, std::ios::in | std::ios::binary);
    quadruplet_prefix_binary.open(quadruplet_prefix_path, std::ios::in | std::ios::binary);


    vector<float> realThreshold;

    vector<float> singleThreshold;
    vector<int> singleEstimatedK;
    vector<int> budgetUsed;
    vector<float> singleQueryTimeVec;

    auto t_start = std::chrono::high_resolution_clock::now();
    auto t_end = std::chrono::high_resolution_clock::now();

    int count = 0;
    for (auto const& query: queries) {

        auto terms = query.terms;
        const int query_length = terms.size();

        topk_queue topk_old(1000);
        wand_query wand_q_old(topk_old);

        // calculate all terms threshold
        wand_q_old(make_max_scored_cursors(index, wdata, *scorer, query), index.num_docs());
        topk_old.finalize();
        auto allTermResults = topk_old.topk();

        topk_old.clear();
        float allTermThreshold = -1.0;
        if (allTermResults.size() >= k) {
            allTermThreshold = allTermResults[k - 1].first;
        }

        realThreshold.push_back(allTermThreshold);

        // If doc less than k
        if (allTermThreshold == -1.0 || query_length > MaxQueryLen) {
            singleThreshold.push_back(-1.0);
            singleEstimatedK.push_back(-1);
            singleQueryTimeVec.push_back(-1.0);
            budgetUsed.push_back(-1);
            continue;
        }

        t_start = std::chrono::high_resolution_clock::now();
        priority_queue<Posting, vector<Posting>, ComparePostingScore> posting_max_heap;
        vector<string> allPossibleComb = getAllPossibleComb(terms, termConsidered);
        float curEstimate;

        // Last one will be the total score
        dense_hash_map<uint64_t, pair<bitset<MaxQueryLen>, vector<short>>, hash<uint64_t>, eq64int> did_scores_map;
        did_scores_map.set_empty_key(numeric_limits<uint64_t>::max());
        dense_hash_map<uint32_t, short, hash<uint32_t>, eq32int> term_position_map;
        term_position_map.set_empty_key(numeric_limits<uint32_t>::max());

        short pos = 0;
        for (uint32_t term : terms) {
            term_position_map[term] = pos;
            pos++;
        }

        //t_start = std::chrono::high_resolution_clock::now();
        for (string comb : allPossibleComb) {
            if (lex_map.find(comb) != lex_map.end()) {
                insert_posting_to_heap(posting_max_heap, comb, 0, lex_map, SampleDid);
            }
        }

        //clog << "initial heap size is: " << posting_max_heap.size() << endl;
        int cur_budget = budget;

        while (cur_budget > 0 && posting_max_heap.size() > 0) {
            Posting top_posting = posting_max_heap.top();
            posting_max_heap.pop();
            int if_push_success = insert_posting_to_heap(posting_max_heap, top_posting.term_string, top_posting.rank + 1, lex_map, SampleDid);

            if (if_push_success == 1) {
                //clog << "push success" << endl;
                cur_budget--;
            } else {
                //clog << "push failed" << endl;
            }

            uint64_t popped_posting_did = top_posting.did;
            vector<uint32_t> popped_posting_term_ids = getTermsFromString(top_posting.term_string);
            vector<short> popped_posting_term_scores = top_posting.scores;

            if (did_scores_map.find(popped_posting_did) == did_scores_map.end()) {
                did_scores_map[popped_posting_did].second.resize(query_length + 1, 0);
            }

            if (did_scores_map[popped_posting_did].first.count() == query_length) {
                continue;
            }
            for (int i = 0; i < popped_posting_term_ids.size(); i++) {
                uint32_t cur_term_id = popped_posting_term_ids[i];
                short cur_term_score = popped_posting_term_scores[i];
                short term_position = term_position_map[cur_term_id];
                if (!did_scores_map[popped_posting_did].first.test(term_position)) {
                    did_scores_map[popped_posting_did].second[term_position] = cur_term_score;
                    did_scores_map[popped_posting_did].second.back() += cur_term_score;
                    did_scores_map[popped_posting_did].first.set(term_position);
                }
            }
            //clog << "did scores map count is: " << did_scores_map.size() << endl;
        }


        if (did_scores_map.size() >= KPrime) {
            vector<short> all_combined_scores;
            for(auto key_scores : did_scores_map) {
                all_combined_scores.push_back(key_scores.second.second.back());
            }

            sort(all_combined_scores.begin(), all_combined_scores.end(), greater<short>());

            curEstimate = all_combined_scores[KPrime - 1];
            //clog << "budget remaining: " << cur_budget << endl;
        } else {
            curEstimate = -2.0;
        }

        t_end = std::chrono::high_resolution_clock::now();
        singleThreshold.push_back(curEstimate);
        singleQueryTimeVec.push_back(std::chrono::duration<double, std::milli>(t_end-t_start).count());

        if (curEstimate < 0) {
            singleEstimatedK.push_back(-2);
            budgetUsed.push_back(-2);
            continue;
        }

        for (int i = 0; i < allTermResults.size() - 1; i++) {
            if (allTermResults[i].first >= curEstimate && allTermResults[i + 1].first <= curEstimate) {
                singleEstimatedK.push_back(i + 2);
                break;
            } else if (i == allTermResults.size() - 2) {
                singleEstimatedK.push_back(i + 2);
                break;
            }
        }

        count++;
        if (count % 10 == 0) {
            clog << count << "queries consider terms = " << termConsidered << " k = " << k << " processed -- combine terms" << endl;
        }
    }

    single_prefix_binary.close();
    duplet_prefix_binary.close();
    triplet_prefix_binary.close();
    quadruplet_prefix_binary.close();

    // print result
    std::cout << "real threshold" << endl;
    std::cout << "*************************************************************" << endl;
    for(int i = 0; i < realThreshold.size(); i++) {
        cout << realThreshold[i] << '\n';
    }

    std::cout << "single threshold" << endl;
    std::cout << "*************************************************************" << endl;
    for(int i = 0; i < singleThreshold.size(); i++) {
        cout << singleThreshold[i] << '\n';
    }
    std::cout << "single estimated K" << endl;
    std::cout << "*************************************************************" << endl;
    for(int i = 0; i < singleEstimatedK.size(); i++) {
        cout << singleEstimatedK[i] << '\n';
    }
    std::cout << "single estimated time" << endl;
    std::cout << "*************************************************************" << endl;
    for(int i = 0; i < singleQueryTimeVec.size(); i++) {
        cout << singleQueryTimeVec[i] << '\n';
    }

    std::cout << "single budget used" << endl;
    std::cout << "*************************************************************" << endl;
    for(int i = 0; i < budgetUsed.size(); i++) {
        cout << budgetUsed[i] << '\n';
    }

}

using wand_raw_index = wand_data<wand_data_raw>;
using wand_uniform_index = wand_data<wand_data_compressed<>>;
using wand_uniform_index_quantized = wand_data<wand_data_compressed<PayloadType::Quantized>>;

int main(int argc, const char** argv)
{
    spdlog::drop("");
    spdlog::set_default_logger(spdlog::stderr_color_mt(""));

    std::string query_filename;
    std::optional<std::string> pairs_filename;
    std::optional<std::string> triples_filename;
    std::optional<std::string> exist_term_filename;
    std::string index_filename;
    std::string wand_data_filename;
    bool quantized = false;

    bool all_pairs = false;
    bool all_triples = false;

    App<arg::Index, arg::WandData<arg::WandMode::Required>, arg::Query<arg::QueryMode::Ranked>, arg::Scorer>
        app{"A tool for performing threshold estimation using the k-highest impact score for each "
            "term, pair or triple of a query. Pairs and triples are only used if provided with "
            "--pairs and --triples respectively."};
    auto pairs = app.add_option(
        "-p,--pairs", pairs_filename, "A tab separated file containing all the cached term pairs");
    auto triples = app.add_option(
        "-t,--triples",
        triples_filename,
        "A tab separated file containing all the cached term triples");
    auto exist_terms = app.add_option(
        "--exist", exist_term_filename, "A newline separated file containing all the exist term");

    app.add_flag("--all-pairs", all_pairs, "Consider all term pairs of a query")->excludes(pairs);
    app.add_flag("--all-triples", all_triples, "Consider all term triples of a query")->excludes(triples);
    app.add_flag("--quantized", quantized, "Quantizes the scores");

    CLI11_PARSE(app, argc, argv);

    auto params = std::make_tuple(
        app.index_filename(),
        app.wand_data_path(),
        app.queries(),
        app.index_encoding(),
        app.scorer_params(),
        app.k(),
        quantized,
        pairs_filename,
        triples_filename,
        exist_term_filename,
        all_pairs,
        all_triples);

    /**/
    if (false) {
#define LOOP_BODY(R, DATA, T)                                                                      \
    }                                                                                              \
    else if (app.index_encoding() == BOOST_PP_STRINGIZE(T))                                        \
    {                                                                                              \
        if (app.is_wand_compressed()) {                                                            \
            if (quantized) {                                                                       \
                std::apply(                                                                        \
                    kt_thresholds<BOOST_PP_CAT(T, _index), wand_uniform_index_quantized>, params); \
            } else {                                                                               \
                std::apply(kt_thresholds<BOOST_PP_CAT(T, _index), wand_uniform_index>, params);    \
            }                                                                                      \
        } else {                                                                                   \
            std::apply(kt_thresholds<BOOST_PP_CAT(T, _index), wand_raw_index>, params);            \
        }
        /**/
        BOOST_PP_SEQ_FOR_EACH(LOOP_BODY, _, PISA_INDEX_TYPES);
#undef LOOP_BODY

    } else {
        spdlog::error("Unknown type {}", app.index_encoding());
    }
}
