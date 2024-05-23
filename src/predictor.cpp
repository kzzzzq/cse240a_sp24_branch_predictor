//========================================================//
//  predictor.c                                           //
//  Source file for the Branch Predictor                  //
//                                                        //
//  Implement the various branch predictors below as      //
//  described in the README                               //
//========================================================//
#include <stdio.h>
#include <math.h>
#include "predictor.h"

//
// TODO:Student Information
//
const char *studentName = "Zheng Kuang";
const char *studentID = "A59023164";
const char *email = "z1kuang@ucsd.edu";

//------------------------------------//
//      Predictor Configuration       //
//------------------------------------//

// Handy Global for use in output routines
const char *bpName[4] = {"Static", "Gshare",
                         "Tournament", "Custom"};

// define number of bits required for indexing the BHT here.
int ghistoryBits = 14; // Number of bits used for Global History
int bpType;            // Branch Prediction Type
int verbose;

//------------------------------------//
//      Predictor Data Structures     //
//------------------------------------//

//
// TODO: Add your own Branch Predictor data structures here
//
// gshare
uint8_t *bht_gshare;
uint64_t ghistory;

//tournament 
int lh_tournament = 10; //local history = 10bits
int ghistoryBits_tournament = 12; //global history = 12 bits
uint16_t *lht_tournament; //lht = local history table, 1024 * 10bits
uint8_t *lPredict_tournament; //local prediction table, 1024 * 2bits
uint8_t *gPredict_tournament; //global prediction table, 4096 * 2bits
uint8_t *choicePrd_tournament; //choice prediction, 4096 * 2bits


//perceptron
int ghistBits_percp = 30; //global history = 30bits, 31 weights per perceptron
int pcBits_percp = 6; //LS 6 bits of PC
int thrd = 49; //threshold for training
int8_t *pt_percp; //table of perceptron, weight is signed 7-bit integer

/*int lhistBits_percp = 0; //local history = 5bits, 6 bits per perceptron
int thrd_l = 0; //threshold for local perceptron
int8_t *pt_percp_l; //table of local perceptron, signed 6-bit integer
*/
int pc_borrowBits = 7; //bits from pc as the inputs to the perceptron

//uint8_t *lht_percp; //local history table for perceptron


int gPredict_percp_Bits = 12; //gshare table size
uint8_t *gPredict_percp;
uint8_t *choicePrd_percp; //choice table for perceptron, 2^12 * 2bits; 00,01 = perceptron, 10,11 = global perdiction


//------------------------------------//
//        Predictor Functions         //
//------------------------------------//
////////////////////perceptron functions: init, dot product, predict, train
void init_perceptron()
{
  int pt_entries = 1 << pcBits_percp; //2^7 entries for perceptron table
  pt_percp = (int8_t *) malloc(pt_entries * (ghistBits_percp + 1 + pc_borrowBits) * sizeof(int8_t)); //perceptron table, 2^7 rows, 21 + 7 columns
  int i,j;
  for (i = 0; i<pt_entries;i++)
  {
    for (j = 0; j<ghistBits_percp + 1 + pc_borrowBits;j++)
    {
      pt_percp[i*(ghistBits_percp + 1 + pc_borrowBits)+j] = (j == 0) ? 1 : 0; //initiallize all weights to 1 except the bias weight
    }
  }
  
  /*pt_percp_l = (int8_t *) malloc(pt_entries * (lhistBits_percp + 1) * sizeof(int8_t)); //perceptron table, 2^7 rows, 6 columns
  for (i = 0; i<pt_entries;i++)
  {
    for (j = 0; j<lhistBits_percp + 1;j++)
    {
      pt_percp_l[i*(lhistBits_percp + 1)+j] = (j == 0) ? 1 : 0; //initiallize all weights to 1 except the bias weight
    }
  }
  */

  /*lht_percp = (uint8_t *) malloc(pt_entries * sizeof(uint8_t));
  for (i = 0; i<pt_entries;i++)
  {
    lht_percp[i] = 0;
  }
  */

  int gc_entries = 1 << gPredict_percp_Bits; //2^11
  gPredict_percp = (uint8_t *) malloc(gc_entries * sizeof(uint8_t));
  choicePrd_percp = (uint8_t *) malloc(gc_entries * sizeof(uint8_t));
  for (i = 0; i<gc_entries;i++)
  {
    gPredict_percp[i] = WN;
    choicePrd_percp[i] = 1;
  }

  ghistory = 0;
}

int * dot_product(uint32_t pc) //yout
{
  uint32_t pt_entries = 1 << pcBits_percp;
  uint32_t pc_lower_bits = pc & (pt_entries - 1); //LS 7 bits of PC
  uint32_t ghist_entries = 1 << ghistBits_percp;
  uint32_t ghistory_lower_bits = ghistory & (ghist_entries - 1); //LS 20 bits of global history

  int8_t *p_selected = (int8_t *) malloc((ghistBits_percp + 1 + pc_borrowBits) * sizeof(int8_t)); //selected perceptron (for global)
  int i;
  for (i = 0; i<ghistBits_percp + 1 + pc_borrowBits;i++)
  {
    p_selected[i] = pt_percp[pc_lower_bits * (ghistBits_percp + 1 + pc_borrowBits) + i];
  }

  /*int8_t *p_selected_l = (int8_t *) malloc((lhistBits_percp + 1) * sizeof(int8_t)); //selected perceptron (for local)
  for (i = 0; i<lhistBits_percp + 1;i++)
  {
    p_selected_l[i] = pt_percp_l[pc_lower_bits * (lhistBits_percp + 1) + i];
  }
*/

  int8_t *x_ghist = (int8_t *) malloc((ghistBits_percp + 1) * sizeof(int8_t)); //input to the perceptron, global history
  for (i = 0; i<ghistBits_percp + 1;i++)
  {
    x_ghist[i] = -1;
  }
  x_ghist[0] = 1; //bias

  uint32_t j = ghistory_lower_bits;
  int n = 1;
  while (j > 0)
  {
    x_ghist[n] = (j%2 == 0) ? -1 : 1; //TAKEN = 1, NOTTAKEN = -1
    j = j/2;
    n++;
  }

  uint32_t pc_shifted = pc >> pcBits_percp;
  uint32_t pc_shifted_lower_bits = pc_shifted & ((1 << pc_borrowBits) - 1);//pc[13:7]
  int8_t *x_pc = (int8_t *) malloc((pc_borrowBits) * sizeof(int8_t)); //input to the perceptron, pc
  for (i = 0; i<pc_borrowBits;i++)
  {
    x_pc[i] = -1;
  }
 //x_ghist[0] = 1; //bias

  j = pc_shifted_lower_bits;
  n = 0;
  while (j > 0)
  {
    x_pc[n] = (j%2 == 0) ? -1 : 1; //TAKEN = 1, NOTTAKEN = -1
    j = j/2;
    n++;
  }

  /*int8_t *x_lhist = (int8_t *) malloc((lhistBits_percp + 1) * sizeof(int8_t)); //input to the perceptron, local history
  for (i = 0; i<lhistBits_percp + 1;i++)
  {
    x_lhist[i] = -1;
  }
  x_lhist[0] = 1; //bias

  uint8_t lh_entries = 1 << lhistBits_percp;
  uint8_t k = lht_percp[pc_lower_bits] & (lh_entries - 1); //local history LS 5 bits
  n = 1;
  while (k > 0)
  {
    x_lhist[n] = (k%2 == 0) ? -1 : 1; //TAKEN = 1, NOTTAKEN = -1
    k = k/2;
    n++;
  }*/


  int *yout = (int *) malloc(1 * sizeof(int)); //prediction given by dot product x and w
  yout[0] = 0;
  //yout[1] = 0;
  for (n=0; n<ghistBits_percp + 1; n++)
  {
    yout[0] += x_ghist[n] * p_selected[n]; //global perceptron
  }
  for (n=0; n<pc_borrowBits; n++)
  {
    yout[0] += x_pc[n] * p_selected[n + ghistBits_percp + 1];
  }
  /*
  for (n=0; n<lhistBits_percp + 1; n++)
  {
    yout[1] += x_lhist[n] * p_selected_l[n];//local perceptron
  }
  */
  return yout;
}

uint8_t perceptron_predict(uint32_t pc)
{
  int *yout = dot_product(pc);
  uint8_t result_percp;
  uint8_t result_g;
  uint8_t choice;
  uint8_t prediction;
  int temp = yout[0] /*+ yout[1]*/;
  result_percp = (temp >= 0) ? TAKEN : NOTTAKEN;

  uint32_t gc_entries = 1 << gPredict_percp_Bits; //2^11
  uint16_t ghistory_lower_bits_global = ghistory & (gc_entries - 1); //LS 11 bits of ghistory
  uint16_t pc_lower_bits_global = pc & (gc_entries - 1); //LS 12 bits of pc
  result_g = gPredict_percp[ghistory_lower_bits_global ^ pc_lower_bits_global];

  choice = choicePrd_percp[ghistory_lower_bits_global];
  if (choice < 2) //choose perceptron
  {
    return result_percp;
  }
  else //choose global
  {
    prediction = (result_g < 2) ? NOTTAKEN : TAKEN;
    return prediction;
  }
}

void train_perceptron(uint32_t pc, uint8_t outcome)
{
  int8_t t = (outcome == TAKEN) ? 1 : -1; //signed outcome

  int *yout = dot_product(pc);
  
  int8_t prd_g = (yout[0] >= 0) ? 1 : -1; //signed prediction (global)
  //int8_t prd_l = (yout[1] >= 0) ? 1 : -1; //signed prediction (local)

  uint8_t prd_percp = ((yout[0] /*+ yout[1]*/) >= 0) ? TAKEN : NOTTAKEN;//prediction from perceptron side
  
  uint32_t pt_entries = 1 << pcBits_percp;
  uint32_t pc_lower_bits = pc & (pt_entries - 1); //LS 7 bits of PC
  uint32_t ghist_entries = 1 << ghistBits_percp;
  uint32_t ghistory_lower_bits = ghistory & (ghist_entries - 1); //LS 20 bits of global history
  uint32_t gc_entries = 1 << gPredict_percp_Bits; //2^11
  uint16_t ghistory_lower_bits_global = ghistory & (gc_entries - 1); //LS 11 bits of ghistory
  uint16_t pc_lower_bits_global = pc & (gc_entries - 1); //LS 12 bits of pc

  uint8_t prd_global = gPredict_percp[ghistory_lower_bits_global ^ pc_lower_bits_global];
  prd_global = (prd_global < 2) ? NOTTAKEN : TAKEN; //prediction from global history side

  int i;
  int8_t *x_ghist = (int8_t *) malloc((ghistBits_percp + 1) * sizeof(int8_t)); //input to the perceptron, global history
  for (i = 0; i<ghistBits_percp + 1;i++)
  {
    x_ghist[i] = -1;
  }
  x_ghist[0] = 1; //bias

  uint32_t j = ghistory_lower_bits;
  int n = 1;
  while (j > 0)
  {
    x_ghist[n] = (j%2 == 0) ? -1 : 1; //TAKEN = 1, NOTTAKEN = -1
    j = j/2;
    n++;
  }



  uint32_t pc_shifted = pc >> pcBits_percp;
  uint32_t pc_shifted_lower_bits = pc_shifted & ((1 << pc_borrowBits) - 1);//pc[13:7]
  int8_t *x_pc = (int8_t *) malloc((pc_borrowBits) * sizeof(int8_t)); //input to the perceptron, pc
  for (i = 0; i<pc_borrowBits;i++)
  {
    x_pc[i] = -1;
  }
 //x_ghist[0] = 1; //bias

  j = pc_shifted_lower_bits;
  n = 0;
  while (j > 0)
  {
    x_pc[n] = (j%2 == 0) ? -1 : 1; //TAKEN = 1, NOTTAKEN = -1
    j = j/2;
    n++;
  }
  /*int8_t *x_lhist = (int8_t *) malloc((lhistBits_percp + 1) * sizeof(int8_t)); //input to the perceptron, local history
  for (i = 0; i<lhistBits_percp + 1;i++)
  {
    x_lhist[i] = -1;
  }
  x_lhist[0] = 1; //bias

  uint8_t lh_entries = 1 << lhistBits_percp;
  uint8_t k = lht_percp[pc_lower_bits] & (lh_entries - 1); //local history LS 5 bits
  n = 1;
  while (k > 0)
  {
    x_lhist[n] = (k%2 == 0) ? -1 : 1; //TAKEN = 1, NOTTAKEN = -1
    k = k/2;
    n++;
  }*/

  //update G perceptron and L perceptron table
  if (prd_g != t || abs(yout[0]) <= thrd)
  {
    for (i = 0; i< ghistBits_percp + 1 + pc_borrowBits;i++)
    {
      if (i < ghistBits_percp + 1)
      {
        pt_percp[pc_lower_bits * (ghistBits_percp + 1 + pc_borrowBits) + i] += t * x_ghist[i];
      }
      else
      {
        pt_percp[pc_lower_bits * (ghistBits_percp + 1 + pc_borrowBits) + i] += t * x_pc[i - ghistBits_percp -1];
      }
      
    }
  }

  /*if (prd_l != t || abs(yout[1]) <= thrd_l)
  {
    for (i = 0; i< lhistBits_percp + 1;i++)
    {
      pt_percp_l[pc_lower_bits * (lhistBits_percp + 1) + i] += t * x_lhist[i];
    }
  }*/

  //update choice table
  if ((prd_global != outcome) && prd_percp == outcome)
  {
    if (choicePrd_percp[ghistory_lower_bits_global] != 0)
    {
      choicePrd_percp[ghistory_lower_bits_global]--;
    }
  }
  else if ((prd_global == outcome) && (prd_percp != outcome))
  {
    if (choicePrd_percp[ghistory_lower_bits_global] != 3)
    {
      choicePrd_percp[ghistory_lower_bits_global]++;
    }
  }

  //update global prediction table
  if (outcome == TAKEN)
  {
    if (gPredict_percp[ghistory_lower_bits_global ^ pc_lower_bits_global] != 3)
    {
      gPredict_percp[ghistory_lower_bits_global ^ pc_lower_bits_global]++;
    }

  }
  else
  {
    if (gPredict_percp[ghistory_lower_bits_global ^ pc_lower_bits_global] != 0)
    {
      gPredict_percp[ghistory_lower_bits_global ^ pc_lower_bits_global]--;
    }
  }
  //lht_percp[pc_lower_bits] = (lht_percp[pc_lower_bits] << 1) | outcome;
  
  ghistory = (ghistory << 1) | outcome;
}


////////////////////perceptron end

////////////////////tournament functions: init, predict, train
void init_tournament()
{
  int lht_entries = 1 << lh_tournament; //2^10, for lht and lPredict
  int gc_entries = 1 << ghistoryBits_tournament; //2^12, for gPredict and choicePrd
  lht_tournament = (uint16_t *) malloc(lht_entries * sizeof(uint16_t));
  lPredict_tournament = (uint8_t *) malloc(lht_entries * sizeof(uint8_t));
  gPredict_tournament = (uint8_t *) malloc(gc_entries * sizeof(uint8_t));
  choicePrd_tournament = (uint8_t *) malloc(gc_entries * sizeof(uint8_t)); //00,01 = local prediction; 10,11 = global prediction

  
  int i = 0;
  for (i = 0; i < lht_entries; i++)
  {
    lht_tournament[i] = 0;
    lPredict_tournament[i] = WN;
  }

  for (i = 0; i < gc_entries; i++)
  {
    gPredict_tournament[i] = WN;
    choicePrd_tournament[i] = 1;
  }
  ghistory = 0;
}

uint8_t tournament_predict(uint32_t pc)
{
  // get lower ghistoryBits of pc
  uint32_t lht_entries = 1 << lh_tournament;
  uint32_t pc_lower_bits = pc & (lht_entries - 1); //LS 10 bits of PC
  uint32_t gc_entries = 1 << ghistoryBits_tournament;
  uint32_t ghistory_lower_bits = ghistory & (gc_entries - 1); //LS 12 bits of global history

  uint8_t prediction; //final prediction


  uint8_t choice = choicePrd_tournament[ghistory_lower_bits];
  if (choice < 2) //choose local prediction
  {
    uint16_t lHistory = lht_tournament[pc_lower_bits] & (lht_entries - 1); //LS 10 bits of local history
    prediction = lPredict_tournament[lHistory];

  }
  else //choose global prediction
  {
    prediction = gPredict_tournament[ghistory_lower_bits];
  }
  switch (prediction)
  {
    case WN:
      return NOTTAKEN;
    case SN:
      return NOTTAKEN;
    case WT:
      return TAKEN;
    case ST:
      return TAKEN;
    default:
      printf("Warning: Undefined state of entry in tournament prediction!\n");
      
      return NOTTAKEN;
  }
  
  
}

void train_tournament(uint32_t pc, uint8_t outcome)
{
  // get lower lh of pc
  uint32_t lht_entries = 1 << lh_tournament;
  uint32_t pc_lower_bits = pc & (lht_entries - 1); //LS 10 bits of PC
  //get lower ghistoryBits of ghistory
  uint32_t gc_entries = 1 << ghistoryBits_tournament;
  uint32_t ghistory_lower_bits = ghistory & (gc_entries - 1); //LS 12 bits of global history

  uint8_t lPrd; //local predictions
  uint8_t gPrd; //global prediction
  
  uint16_t lHistory = lht_tournament[pc_lower_bits] & (lht_entries - 1); //LS 10 bits of local history
  lPrd = lPredict_tournament[lHistory];
  if (lPrd < 2)
  {
    lPrd = NOTTAKEN;
  }
  else
  {
    lPrd = TAKEN;
  }

  gPrd = gPredict_tournament[ghistory_lower_bits];
  if (gPrd < 2)
  {
    gPrd = NOTTAKEN;
  }
  else
  {
    gPrd = TAKEN;
  }
  

  //update choice table
  if (lPrd != outcome && gPrd == outcome) //local wrong, global correct
  {
    if (choicePrd_tournament[ghistory_lower_bits] != 3)
    {
      choicePrd_tournament[ghistory_lower_bits]++;
    }
  }
  else if (lPrd == outcome && gPrd != outcome) //local correct, global wrong
  {
    if (choicePrd_tournament[ghistory_lower_bits] != 0)
    {
      choicePrd_tournament[ghistory_lower_bits]--;
    }
  }


  //update global and local prediction table
  if (outcome == NOTTAKEN)
  {
    if (lPredict_tournament[lHistory] != 0)
    {
      lPredict_tournament[lHistory]--;
    }
    if (gPredict_tournament[ghistory_lower_bits] != 0)
    {
      gPredict_tournament[ghistory_lower_bits]--;
    }
  }
  else
  {
    if (lPredict_tournament[lHistory] != 3)
    {
      lPredict_tournament[lHistory]++;
    }
    if (gPredict_tournament[ghistory_lower_bits] != 3)
    {
      gPredict_tournament[ghistory_lower_bits]++;
    }
  }

  //update local history and global history
  lht_tournament[pc_lower_bits] = (lht_tournament[pc_lower_bits] << 1) | outcome;
  ghistory = (ghistory << 1) | outcome;
}

////////////////// tournament end

// Initialize the predictor
//

// gshare functions
void init_gshare()
{
  int bht_entries = 1 << ghistoryBits;
  bht_gshare = (uint8_t *)malloc(bht_entries * sizeof(uint8_t));
  int i = 0;
  for (i = 0; i < bht_entries; i++)
  {
    bht_gshare[i] = WN;
  }
  ghistory = 0;
}


uint8_t gshare_predict(uint32_t pc)
{
  // get lower ghistoryBits of pc
  uint32_t bht_entries = 1 << ghistoryBits;
  uint32_t pc_lower_bits = pc & (bht_entries - 1);
  uint32_t ghistory_lower_bits = ghistory & (bht_entries - 1);
  uint32_t index = pc_lower_bits ^ ghistory_lower_bits;
  switch (bht_gshare[index])
  {
  case WN:
    return NOTTAKEN;
  case SN:
    return NOTTAKEN;
  case WT:
    return TAKEN;
  case ST:
    return TAKEN;
  default:
    printf("Warning: Undefined state of entry in GSHARE BHT!\n");
    return NOTTAKEN;
  }
}

void train_gshare(uint32_t pc, uint8_t outcome)
{
  // get lower ghistoryBits of pc
  uint32_t bht_entries = 1 << ghistoryBits;
  uint32_t pc_lower_bits = pc & (bht_entries - 1);
  uint32_t ghistory_lower_bits = ghistory & (bht_entries - 1);
  uint32_t index = pc_lower_bits ^ ghistory_lower_bits;

  // Update state of entry in bht based on outcome
  switch (bht_gshare[index])
  {
  case WN:
    bht_gshare[index] = (outcome == TAKEN) ? WT : SN;
    break;
  case SN:
    bht_gshare[index] = (outcome == TAKEN) ? WN : SN;
    break;
  case WT:
    bht_gshare[index] = (outcome == TAKEN) ? ST : WN;
    break;
  case ST:
    bht_gshare[index] = (outcome == TAKEN) ? ST : WT;
    break;
  default:
    printf("Warning: Undefined state of entry in GSHARE BHT!\n");
    break;
  }

  // Update history register
  ghistory = ((ghistory << 1) | outcome);
}

void cleanup_gshare()
{
  free(bht_gshare);
}

void init_predictor()
{
  switch (bpType)
  {
  case STATIC:
    break;
  case GSHARE:
    init_gshare();
    break;
  case TOURNAMENT:
    init_tournament();
    break;
  case CUSTOM:
    init_perceptron();
    break;
  default:
    break;
  }
}

// Make a prediction for conditional branch instruction at PC 'pc'
// Returning TAKEN indicates a prediction of taken; returning NOTTAKEN
// indicates a prediction of not taken
//
uint32_t make_prediction(uint32_t pc, uint32_t target, uint32_t direct)
{

  // Make a prediction based on the bpType
  switch (bpType)
  {
  case STATIC:
    return TAKEN;
  case GSHARE:
    return gshare_predict(pc);
  case TOURNAMENT:
    return tournament_predict(pc);
  case CUSTOM:
    return perceptron_predict(pc);
  default:
    break;
  }

  // If there is not a compatable bpType then return NOTTAKEN
  return NOTTAKEN;
}

// Train the predictor the last executed branch at PC 'pc' and with
// outcome 'outcome' (true indicates that the branch was taken, false
// indicates that the branch was not taken)
//

void train_predictor(uint32_t pc, uint32_t target, uint32_t outcome, uint32_t condition, uint32_t call, uint32_t ret, uint32_t direct)
{
  if (condition)
  {
    switch (bpType)
    {
    case STATIC:
      return;
    case GSHARE:
      return train_gshare(pc, outcome);
    case TOURNAMENT:
      return train_tournament(pc, outcome);
    case CUSTOM:
      return train_perceptron(pc, outcome);
    default:
      break;
    }
  }
}
