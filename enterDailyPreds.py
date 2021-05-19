
#-- import base packages
import time
import re
#-- import pypi packages
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

#-- import local python functions


def enter(pick1, pick2=None):
    #---- define variables
    print('loading chrome driver....')
    driver = webdriver.Chrome('./chromedriver')
    url = 'https://bts-web.mlb.com/mlb/fantasy/bts/y2021/index.jsp#t=picks'
    email = 'fake_email@gmail.com'
    pw = 'thisisnotmypassword'
    todays_pick_reactid = "'.0.1.1.1.0.0'"
    todays_pick_reactid2 = todays_pick_reactid[:12] + '1' +todays_pick_reactid[13:]
    select_team_reactid = "'.0.1.1.1.0.1.0.0.0.4'"
    select_team_reactid2 = select_team_reactid[:12] + '2' +select_team_reactid[13:]
    tbody_reactid = "'.0.1.1.1.0.1.1.0.1'"
    select_suffix = ".3.0"
    player_id_search_pattern = "people/(.*?)/headshot"
    team_dict = {}

    ids = [
        "'.0.1.1.1.0.1.0.2.0.0'", "'.0.1.1.1.0.1.0.2.0.1'",
        "'.0.1.1.1.0.1.0.2.0.2'", "'.0.1.1.1.0.1.0.2.0.3'",
        "'.0.1.1.1.0.1.0.2.0.4'", "'.0.1.1.1.0.1.0.2.0.5'",
        "'.0.1.1.1.0.1.0.2.0.6'", "'.0.1.1.1.0.1.0.2.0.7'",
        "'.0.1.1.1.0.1.0.2.0.8'", "'.0.1.1.1.0.1.0.2.0.a'",
        "'.0.1.1.1.0.1.0.2.0.b'", "'.0.1.1.1.0.1.0.2.0.c'",
        "'.0.1.1.1.0.1.0.2.0.d'", "'.0.1.1.1.0.1.0.2.0.e'",
        "'.0.1.1.1.0.1.0.2.0.f'", "'.0.1.1.1.0.1.0.2.0.g'",
        "'.0.1.1.1.0.1.0.2.0.h'", "'.0.1.1.1.0.1.0.2.0.i'",
        "'.0.1.1.1.0.1.0.2.0.j'", "'.0.1.1.1.0.1.0.2.0.k'",
        "'.0.1.1.1.0.1.0.2.0.l'", "'.0.1.1.1.0.1.0.2.0.m'",
        "'.0.1.1.1.0.1.0.2.0.n'", "'.0.1.1.1.0.1.0.2.0.o'",
        "'.0.1.1.1.0.1.0.2.0.p'", "'.0.1.1.1.0.1.0.2.0.q'",
        "'.0.1.1.1.0.1.0.2.0.r'", "'.0.1.1.1.0.1.0.2.0.s'",
        "'.0.1.1.1.0.1.0.2.0.t'"
    ]


    #-- Run Selenium Processes
    print('loading url....')
    driver.get(url)
    time.sleep(15)
    print('selecting log in....')
    driver.find_element_by_class_name('ui-dialog-buttonset').click()
    print('logging in....')
    driver.find_element_by_id('okta-signin-username').send_keys(email)
    driver.find_element_by_id('okta-signin-password').send_keys(pw)
    driver.find_element_by_id("okta-signin-submit").click()
    time.sleep(15)
    #---- Pick #1
    #-- todays pick #1
    print('opening team selection and collecting react_ids....')
    driver.find_element_by_xpath("(//div[@data-reactid={}])".format(todays_pick_reactid)).click()
    driver.find_element_by_xpath("(//li[@data-reactid={}])".format(select_team_reactid)).click()
    time.sleep(3)

    #-- pick by team
    for id in ids:
        try:
            id_fixed = id#.replace("'.0.1.1", "'.0.1.2") # temporary for dev
            el = driver.find_element_by_xpath("(//li[@data-reactid={}])".format(id_fixed))
            team = el.get_attribute("class").split(' ')[0].upper()
            team_dict[team] = {}
            team_dict[team]['team_reactid'] = id_fixed

        except Exception as e:
            print('error on id: {}'.format(id_fixed))
            pass
    cols = ['team', 'team_reactid', 'player_name', 'player_reactid', 'player_id']
    player_df = pd.DataFrame(columns=cols)

    driver.find_element_by_xpath("(//li[@data-reactid={}])".format(select_team_reactid)).click()
    #-- create player_df
    print('iterating through teams to collect player react_ids....')
    for team in team_dict.keys():
        driver.find_element_by_xpath("(//li[@data-reactid={}])".format(select_team_reactid)).click()
        time.sleep(3)
        driver.find_element_by_xpath("(//li[@data-reactid={}])".format(team_dict[team]['team_reactid'])).click()
        time.sleep(3)
        tbody = driver.find_element_by_xpath("(//tbody[@data-reactid={}])".format(tbody_reactid))
        rows = tbody.find_elements_by_tag_name("tr")
        players = []
        for r in rows:
            im = r.find_element_by_tag_name("td").find_element_by_tag_name("img")
            src = im.get_attribute("src")
            reactid = im.get_attribute("data-reactid")
            #players.append((re.search(player_id_search_pattern, reactid).group(1), reactid))
            #team_dict[team]['players'] = players
            df_row = {
                'team': team,
                'team_reactid': team_dict[team]['team_reactid'],
                'player_name': '',
                'player_reactid': reactid[:-4],
                'player_id': re.search(player_id_search_pattern, src).group(1)
            }
            player_df = player_df.append(df_row, ignore_index=True)
    player_df.to_csv('bts_player_df.csv', index=False)

    #--- submit picks
    print('submitting picks....')

    player_row = player_df.loc[player_df['player_id']==pick1]
    player_team_reactid = player_row['team_reactid'].iloc[0]
    player_reactid = player_row['player_reactid'].iloc[0]
    driver.find_element_by_xpath("(//li[@data-reactid={}])".format(select_team_reactid)).click()
    time.sleep(3)
    driver.find_element_by_xpath("(//li[@data-reactid={}])".format(player_team_reactid)).click()
    time.sleep(3)
    driver.find_element_by_xpath("(//button[@data-reactid='{}{}'])".format(player_reactid, select_suffix)).click()
    if pick2 != None:
        print('submitting pick #2...')
        time.sleep(3)
        driver.find_element_by_xpath("(//div[@data-reactid={}])".format(todays_pick_reactid2)).click()
        player_row = player_df.loc[player_df['player_id']==pick2]
        player_team_reactid = player_row['team_reactid'].iloc[0]
        print(player_team_reactid)
        player_team_reactid = player_team_reactid[:12] + '2' +player_team_reactid[13:]
        #player_team_reactid = select_team_reactid2[0:-1]+player_team_reactid[len(select_team_reactid2)-1:]
        print(player_team_reactid)
        player_reactid = "'" + player_row['player_reactid'].iloc[0]
        player_reactid = player_reactid[0:12]+'2'+player_reactid[13:]
        print(player_reactid)
        driver.find_element_by_xpath("(//li[@data-reactid={}])".format(select_team_reactid2)).click()
        time.sleep(3)
        driver.find_element_by_xpath("(//li[@data-reactid={}])".format(player_team_reactid)).click()
        time.sleep(3)
        driver.find_element_by_xpath("(//button[@data-reactid={}{}])".format(player_reactid, select_suffix+"'")).click()
    driver.quit()
    return player_df


if __name__ == '__main__':
    #-- test with Rafael Devers,  Mookie
    enter("646240", "605141")
