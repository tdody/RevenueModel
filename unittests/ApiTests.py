import urllib.request  as urllib2 
from flask import Flask
from flask_testing import LiveServerTestCase
import unittest
from app import app

# Testing with LiveServer
class ApiTest(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    def test_main(self):
        rv = self.app.get('/')
        assert rv.status == '200 OK'

    def test_train(self):
        rv = self.app.get('/train')
        assert rv.status == '200 OK'

    def test_predict(self):
        rv = self.app.get('/predict')
        assert rv.status == '200 OK'